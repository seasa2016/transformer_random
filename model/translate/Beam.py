import torch
import torch.nn.functional as F
from . import penalties

class beam(object):
	"""
	Class for managing the internals of the beam search process.

	Takes care of beams, back pointers, and scores.

	Args:
	   size (int): beam size
	   pad, bos, eos (int): indices of padding, beginning, and ending.
	   n_best (int): nbest size to use
	   cuda (bool): use gpu
	   global_scorer (:obj:`GlobalScorer`)
	"""

	def __init__(self,size,pad,bos,eos,
				n_best=1,cuda=False,
				global_scorer=None,
				min_length=0,stepwise_penalty=False,
				block_ngram_repeat=0,
				exclusion_tokens=set()):

		self.size = size

		#this is for short variable setting
		device = torch.device('cuda' if cuda else 'cpu')

		self.scores = torch.FloatTensor(size).zero_().to(device)
		self.all_scores = []

		self.prev_ks = []

		self.next_ys = [ torch.LongTensor(size).fill_(pad).to(device)]
		self.next_ys[0][0] = bos
		
		self._eos = eos
		self.eos_top = False
		
		self.attn = []

		self.finished = []
		self.n_best = n_best

		self.global_scorer = global_scorer
		self.global_state = {}

		self.min_length = min_length

		self.stepwise_penalty = stepwise_penalty
		self.block_ngram_repeat = block_ngram_repeat
		self.exclusion_tokens = exclusion_tokens
	
	def get_current_state(self):
		"get the output for the curent timestep"
		return self.next_ys[-1]
	
	def get_current_origin(self):
		"Get the backpointers for the current timestep."
		return self.prev_ks[-1]

	def advance(self,word_probs,attn_out):
		"""
		Given prob over words for every last beam `wordLk` and attention
		`attn_out`: Compute and update the beam search.

		Parameters:

		* `word_probs`- probs of advancing from the last step (K x words)
		* `attn_out`- attention at the last step

		Returns: True if beam search is complete.
		"""
		#print("self._eos",self._eos)
		#print("word_probs",word_probs.shape)
		word_probs = F.log_softmax(word_probs,dim=-1)
		num_words = word_probs.shape[1]
		if(self.stepwise_penalty):
			self.global_scorer.update_score(self,attn_out)
		
		#force the output to be longer than self.min
		cur_len = len(self.next_ys)
		if(cur_len < self.min_length):
			for k in range(word_probs.shape[0]):
				word_probs[k][self._eos] = -1e20
		 
		#sum previous scores.
		#for temp in self.next_ys:
		#	print('-',temp)
		#print('-'*10)
		#for temp in self.prev_ks:
		#	print(temp)
		#print('-'*10)
		if(len(self.prev_ks)>0):
			beam_scores = word_probs + self.scores.unsqueeze(1).expand_as(word_probs)
			#end at eos
			for i in range( self.next_ys[-1].shape[0] ):
				if(self.next_ys[-1][i] == self._eos):
					beam_scores[i] = -1e20
			
			#block ngram repeats
			if(self.block_ngram_repeat>0):
				#here we dont want to generate the same pattern
				#rather than stop at that time 
				
				le = len(self.next_ys)
				if(le >= self.block_ngram_repeat):
					for j in range(self.next_ys[-1].shape[0]):
						hyp,_ = self.get_hyp(le-1,j)

						check = hyp[:self.block_ngram_repeat-1]

						gram = []

						for i in range(le-1):
							#last n tokens, n = block_ngram_repeat
							gram = ( gram + [hyp[i].item()] )[-self.block_ngram_repeat:]

							if(set(gram)&self.exclusion_tokens):
								continue

							if(gram[:self.block_ngram_repeat-1] == check):
								beam_scores[j, gram[-1] ] = -1e20
		else:
			beam_scores = word_probs[0]
		flat_beam_scores = beam_scores.view(-1)
		best_scores,best_scores_id = flat_beam_scores.topk(self.size,0,True,True)
		#print("beat",best_scores,best_scores_id)
		
		self.all_scores.append(self.scores)
		self.scores = best_scores

		#best_scores_id is flattened beam x word array, so calculate which
		#word and beam each score came from
		prev_k = best_scores_id // num_words
		self.prev_ks.append(prev_k)
		self.next_ys.append((best_scores_id - prev_k * num_words))
		
		self.attn.append(attn_out.index_select(0,prev_k))
		self.global_scorer.update_global_state(self)

		for i in range(self.next_ys[-1].shape[0]):
			if(self.next_ys[-1][i] == self._eos):
				global_scores = self.global_scorer.score(self,self.scores)
				s = global_scores[i]
				self.finished.append((s,len(self.next_ys)-1,i))
		
		if(self.next_ys[-1][0] == self._eos):
			self.all_scores.append(self.scores)
			self.eos_top = True

	def done(self):
		return self.eos_top and len(self.finished) >= self.n_best
	
	def sort_finished(self,minium = None):
		if(minium is not None):
			i = 0
			#add from beam until we have minimum outputs
			while(len(self.finished) < minium):
				global_scores = self.global_scorer.score(self,self.scores)
				s = global_scores[i]

				self.finished.append((s,len(self.next_ys)-1,i))
				i += 1
		########################################################
		self.finished.sort(key=lambda a:-a[0])
		scores = [sc for sc,_,_ in self.finished]
		ks = [(t,k) for _,t,k in self.finished]

		return scores,ks
	
	def get_hyp(self,timestep,k):
		"""
		Walk back to construct the full hypothesis.
		"""
		hyp,attn = [],[]
		for j in range(len(self.prev_ks[:timestep])-1,-1,-1):
			
			hyp.append(self.next_ys[j+1][k])
			attn.append(self.attn[j][k])

			k = self.prev_ks[j][k]

		return hyp[::-1], torch.stack(attn[::-1])
	

class GNMTGlobalScorer(object):
	"""
	NMT re-ranking score from
	"Google's Neural Machine Translation System" :cite:`wu2016google`

	Args:
	   alpha (float): length parameter
	   beta (float):  coverage parameter
	"""

	def __init__(self,alpha,beta,cov_penalty,length_penalty):
		self.alpha = alpha
		self.beta = beta
		penalty_builder = penalties.PenaltyBuilder(length_penalty, cov_penalty)

		# Term will be subtracted from probability
		self.cov_penalty = penalty_builder.coverage_penalty()
		# Probability will be divided by this
		self.length_penalty = penalty_builder.length_penalty()
	
	def score(self,beam,logprobs):
		"""
		rescores a prediction based on penalty functions
		"""
		normalized_probs = self.length_penalty(beam,
												logprobs,
												self.alpha)
		if(not beam.stepwise_penalty):
			penalty = self.cov_penalty(beam,
										beam.global_state["coverage"],
										self.beta)
			normalized_probs -= penalty
		
		return normalized_probs

	def update_score(self,beam,attn):
		"""
		function to update scores of a beam that is not finished
		"""
		if("prev_penalty" in beam.global_state.key()):
			beam.scores.add_(beam.global_state["prev_penalty"])
			penalty = self.cov_penalty(beam,
										beam.global_state["coverage"]+attn,
										self.beta)

			beam.scores.sub_(penalty)
	
	def update_global_state(self,beam):
		"keep the coverage vector as sum of attentions"
		if(len(beam.prev_ks) == 1):
			beam.global_state["prev_penalty"] = beam.scores.clone().fill_(0.0)
			beam.global_state['coverage'] = beam.attn[-1]
			self.cov_total = beam.attn[-1].sum(1)
		else:
			self.cov_total += torch.min(beam.attn[-1],
										beam.global_state['coverage']).sum(1)
			beam.global_state["coverage"] = beam.global_state["coverage"] \
				.index_select(0, beam.prev_ks[-1]).add(beam.attn[-1])

			prev_penalty = self.cov_penalty(beam,
											beam.global_state["coverage"],
											self.beta)
			beam.global_state["prev_penalty"] = prev_penalty
			






