import argparse
import codecs
import os
import math
import torch.nn.functional as F
import torch
import sys
#dataset
from data.dataloader import itemDataset,ToTensor,collate_fn
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms, utils


from itertools import count
from ..util.misc import tile

from ..model_builder import load_test_model
import model.translate.Beam as Beam
import model.Constant as Constant
from . import translation
from model.util.Logger import logger

#import data.dataloader
#this should be replace by other dataset

import opts

def to_gpu(batch,cuda):
    if(cuda):
        for name in ['source','target']:
            if(name in batch):
                batch[name] = batch[name].cuda()
    return batch

def build_translator(opt,report_score=True,logger=None,out_file=None):
    """
        build translator for testing
    """
    #This is for the output file for the mathine translation
    if(out_file is None):
        out_file = codecs.open(opt.output,'w+','utf-8')
    
    dummy_parser = argparse.ArgumentParser(description='train.py')
    opts.model_opts(dummy_parser)

    #we wont use ensemble here XD
    model, model_opt = load_test_model(opt)
    
    scorer = Beam.GNMTGlobalScorer( opt.alpha,
                                    opt.beta,
                                    opt.coverage_penalty,
                                    opt.length_penalty)

    kwargs = {k:getattr(opt,k)
              for k in ["batch_size","beam_size", "n_best", "max_length", "min_length",
                        "stepwise_penalty", "block_ngram_repeat",
                        "ignore_when_blocking", "dump_beam", "report_bleu",
                        "replace_unk", "gpu", "verbose", "fast"]}
    

    logger.info("start build testing data")
    #we should deal with the data loader here

    translator = Translator(model,global_scorer=scorer,
                            out_file=out_file, report_score=report_score,
                            copy_attn=model_opt.copy_attn, logger=logger,
                            **kwargs)

    return translator

class Translator(object):
    """
    Uses a model to translate a batch of sentences.


    Args:
       model (:obj:`onmt.modules.NMTModel`):
          NMT model to use for translation
       dataloader: data loader
       beam_size (int): size of beam to use
       n_best (int): number of translations produced
       max_length (int): maximum length output to produce
       global_scores (:obj:`GlobalScorer`):
         object to rescore final translations
       copy_attn (bool): use copy attention during translation
                        I think it will not be use in this work
       cuda (bool): use cuda
       beam_trace (bool): trace beam search for debugging
       logger(logging.Logger): logger.
    """

    def __init__(self,
                 model,
                 batch_size,
                 beam_size,
                 n_best=1,
                 max_length=100,
                 global_scorer=None,
                 copy_attn=False,
                 logger=None,
                 gpu=False,
                 dump_beam="",
                 min_length=0,
                 stepwise_penalty=False,
                 block_ngram_repeat=0,
                 ignore_when_blocking=[],
                 replace_unk=False,
                 report_score=True,
                 report_bleu=False,
                 report_rouge=False,
                 verbose=False,
                 out_file=None,
                 fast=False):
        self.logger = logger
        self.gpu = gpu
        self.cuda = gpu > -1

        self.model = model

        self.batch_size = batch_size
        self.n_best = n_best
        self.max_length = max_length
        self.global_scorer = global_scorer
        self.copy_attn = copy_attn
        self.beam_size = beam_size
        self.min_length = min_length
        self.stepwise_penalty = stepwise_penalty
        self.dump_beam = dump_beam
        self.block_ngram_repeat = block_ngram_repeat
        self.ignore_when_blocking = set(ignore_when_blocking)
        self.replace_unk = replace_unk
        self.verbose = verbose
        self.out_file = out_file
        self.report_score = report_score
        self.report_bleu = report_bleu
        self.report_rouge = report_rouge
        self.fast = fast

        self.vocab = {}
        #set up the vocab for decode
        self.vocab['target'] = {}
        with open('./data/subword.target') as f:
            for i,word in enumerate(f):
                word = word.strip()
                self.vocab['target'][ word[1:-1] ] = i

        # for debugging
        self.beam_trace = self.dump_beam != ""
        self.beam_accum = None
        if self.beam_trace:
            self.beam_accum = {
                "predicted_ids": [],
                "beam_parent_ids": [],
                "scores": [],
                "log_probs": []}


    def translate(self,
                    src_path=None,
                    tgt_path=None,
                    src_data_iter=None,
                    tgt_data_iter=None,
                    attn_debug=False):
        """
        Translate content of `src_data_iter` (if not None) or `src_path`
        and get gold scores if one of `tgt_data_iter` or `tgt_path` is set.

        Note: batch_size must not be None
        Note: one of ('src_path', 'src_data_iter') must not be None

        Args:
            src_path (str): filepath of source data
            e.g. it may be a list or an openned file
            tgt_path (str): filepath of target data
            src_data_iter (iterator): an interator generating source data
            tgt_data_iter (iterator): an interator generating target data
            (used for Audio and Image datasets)
            batch_size (int): size of examples per mini-batch
            attn_debug (bool): enables the attention logging

        Returns:
            (`list`, `list`)

            * all_scores is a list of `batch_size` lists of `n_best` scores
            * all_predictions is a list of `batch_size` lists
            of `n_best` predictions
        """
        assert src_data_iter is not None or src_path is not None

        if(self.batch_size is None):
            raise ValueError("batch size must be set")
        
        #we should build the data loader here

        device = torch.device("cuda" if self.cuda else "cpu")

    
        #feature_type = ['source','target']
        #data_size = dict()
        #for name in feature_type:
        #   with open('./ch_en/subword.{0}'.format(name)) as f_in:
        #       data_size[name] = len(f_in.readlines())
        
        
        test_dataset = itemDataset(file_source=src_path,file_target=tgt_path,transform=transforms.Compose([ToTensor()]))
        self.dataloader = DataLoader(test_dataset, batch_size=self.batch_size,shuffle=False, num_workers=10,collate_fn=collate_fn)
            


        builder  = translation.TranslationBuilder(
            self.n_best, self.replace_unk,tgt_path is not None
        )

        counter = count(1)
        pred_score_total, pred_words_total = 0,0
        gold_score_total, gold_words_total = 0,0

        all_scores = []
        all_predictions = []

        for num,batch in enumerate(self.dataloader):
            print("num",num)
            batch = to_gpu(batch,device)
            batch_data = self.translate_batch(batch,fast=self.fast)
            translations = builder.from_batch(batch_data,num)

            for trans in translations:
                all_scores += [trans.pred_scores[:self.n_best]]
                pred_score_total += trans.pred_scores[0]
                pred_words_total += len(trans.pred_sent[0])
                if tgt_path is not None:
                    gold_score_total += trans.gold_score
                    gold_words_total += len(trans.gold_sent) + 1

                n_best_preds = [pred 
                                for pred in trans.pred_sent]
                all_predictions += [n_best_preds]

                self.out_file.write('\n'.join(n_best_preds)+'\n')
                self.out_file.flush()

                if self.verbose:
                    sent_number = next(counter)
                    output = trans.log(sent_number)
                    if self.logger:
                        self.logger.info(output)
                    else:
                        os.write(1, output.encode('utf-8'))

                # Debug attention.
                if attn_debug:
                    srcs = trans.src_raw
                    preds = trans.pred_sents[0]
                    preds.append('<EOS>')
                    attns = trans.attns[0].tolist()
                    header_format = "{:>10.10} " + "{:>10.7} " * len(srcs)
                    row_format = "{:>10.10} " + "{:>10.7f} " * len(srcs)
                    output = header_format.format("", *trans.src_raw) + '\n'
                    for word, row in zip(preds, attns):
                        max_index = row.index(max(row))
                        row_format = row_format.replace(
                            "{:>10.7f} ", "{:*>10.7f} ", max_index + 1)
                        row_format = row_format.replace(
                            "{:*>10.7f} ", "{:>10.7f} ", max_index)
                        output += row_format.format(word, *row) + '\n'
                        row_format = "{:>10.10} " + "{:>10.7f} " * len(srcs)
                    os.write(1, output.encode('utf-8'))

        if self.report_score:
            msg = self._report_score('PRED', pred_score_total,
                                     pred_words_total)
            if self.logger:
                self.logger.info(msg)
            else:
                print(msg)
            if tgt_path is not None:
                msg = self._report_score('GOLD', gold_score_total,
                                         gold_words_total)
                if self.logger:
                    self.logger.info(msg)
                else:
                    print(msg)
                if self.report_bleu:
                    msg = self._report_bleu(tgt_path)
                    if self.logger:
                        self.logger.info(msg)
                    else:
                        print(msg)
                if self.report_rouge:
                    msg = self._report_rouge(tgt_path)
                    if self.logger:
                        self.logger.info(msg)
                    else:
                        print(msg)

        return all_scores, all_predictions

            

    def translate_batch(self,batch , fast=False):
        """
        Translate a batch of sentences.

        Mostly a wrapper around :obj:`Beam`.

        Args:
           batch (:obj:`Batch`): a batch from a dataset object
           data (:obj:`Dataset`): the dataset object
           fast (bool): enables fast beam search (may not support all features)
        """
        with torch.no_grad():
            if(fast):
                return self._fast_translate_batch(
                    batch,
                    self.max_length,
                    min_length = self.min_length,
                    n_best = self.n_best,
                    return_attention = self.replace_unk
                )
            else:
                return self._translate_batch(batch)
    
    def _fast_translate_batch(self,
                              batch,
                              max_length,
                              min_length=0,
                              n_best=1,
                              return_attention=False):
        # TODO: faster code path for beam_size == 1.
        # TODO: support these blacklisted features.

        #assert not self.copy_attn,'self.copy_attn cannot be {0}'.format(self.copy_attn)
        assert not self.dump_beam
        assert self.block_ngram_repeat == 0
        assert self.global_scorer.beta == 0
        
        beam_size = self.beam_size
        batch_size = batch['source'].shape[1]

        vocab = self.vocab['target']
              
        start_token = vocab[Constant.SOS_token]
        end_token = vocab[Constant.EOS_token]

        src_len = batch["source_len"]
        enc_states, memory_bank,_ = self.model.encoder(batch['source'],src_len)
        dec_states = self.model.decoder.init_decoder_state(
            batch['source'],memory_bank,enc_states,with_cache=True
        )

        dec_states.map_batch_fn(
            lambda state,dim:tile(state,beam_size,dim=dim)
        )

        memory_bank = tile(memory_bank,beam_size,dim=1)
        memory_length = tile(torch.tensor(src_len).to(memory_bank.device),beam_size)

        batch_offset = torch.arange(
            batch_size, dtype=torch.long, device=memory_bank.device)
        beam_offset = torch.arange(
            0,
            batch_size * beam_size,
            step=beam_size,
            dtype=torch.long,
            device=memory_bank.device)
        alive_seq = torch.full(
            [batch_size * beam_size, 1],
            start_token,
            dtype=torch.long,
            device=memory_bank.device)
        alive_attn = None
        
        # Give full probability to the first beam on the first step.
        topk_log_probs = (
            torch.tensor([0.0] + [float("-inf")] * (beam_size - 1),
                         device=memory_bank.device).repeat(batch_size))

        # Structure that holds finished hypotheses.
        hypotheses = [[] for _ in range(batch_size)]  # noqa: F812

        results = {}
        results["predictions"] = [[] for _ in range(batch_size)]  # noqa: F812
        results["scores"] = [[] for _ in range(batch_size)]  # noqa: F812
        results["attention"] = [[] for _ in range(batch_size)]  # noqa: F812
        results["gold_score"] = [0] * batch_size
        results["batch"] = batch

        for step in range(max_length):
            print()
            print("*"*10)
            print("batch_offset",batch_offset)
            print("beam_offset",beam_offset)
            print("alive_seq",alive_seq)
            print("alive_attn",alive_attn)
            print("results",results)
            print("topk_log_probs",topk_log_probs)

            decoder_input = alive_seq[:, -1].view(1, -1)

            # Decoder forward.
            dec_out, dec_states, attn = self.model.decoder(
                decoder_input,
                memory_bank,
                dec_states,
                memory_length=memory_length,
                step=step)
            #print('dec_out',dec_out.shape)
            log_probs = F.log_softmax(dec_out,dim=-1).squeeze(0)
            #print('log_probs',log_probs.shape)
            #print(log_probs.shape)
            #print(log_probs[0][0][:100])
            
            #print("*"*10)
            #print()
            
            vocab_size = log_probs.shape[-1]

            if step < min_length:
                log_probs[:, end_token] = -1e20

            # Multiply probs by the beam probability.
            log_probs += topk_log_probs.view(-1).unsqueeze(1)

            alpha = self.global_scorer.alpha
            length_penalty = ((5.0 + (step + 1)) / 6.0) ** alpha

            # Flatten probs into a list of possibilities.
            curr_scores = log_probs / length_penalty
            curr_scores = curr_scores.reshape(-1, beam_size * vocab_size)
            topk_scores, topk_ids = curr_scores.topk(beam_size, dim=-1)

            # Recover log probs.
            topk_log_probs = topk_scores * length_penalty

            # Resolve beam origin and true word ids.
            topk_beam_index = topk_ids.div(vocab_size)
            topk_ids = topk_ids.fmod(vocab_size)

            # Map beam_index to batch_index in the flat representation.
            batch_index = (
                topk_beam_index
                + beam_offset[:topk_beam_index.shape[0]].unsqueeze(1))
            print("topk_beam_index",topk_beam_index)
            print("beam_offset",beam_offset)
            print("batch_index",batch_index)

            select_indices = batch_index.view(-1)

            # Append last prediction.
            alive_seq = torch.cat(
                [alive_seq.index_select(0, select_indices),
                 topk_ids.view(-1, 1)], -1)
            if return_attention:
                current_attn = attn["std"].index_select(1, select_indices)
                if alive_attn is None:
                    alive_attn = current_attn
                else:
                    alive_attn = alive_attn.index_select(1, select_indices)
                    alive_attn = torch.cat([alive_attn, current_attn], 0)

            is_finished = topk_ids.eq(end_token)
            if step + 1 == max_length:
                is_finished.fill_(1)
            # End condition is top beam is finished.
            ###################################################
            end_condition = is_finished[:, 0].eq(1)

            # Save finished hypotheses.
            if is_finished.any():
                predictions = alive_seq.view(-1, beam_size, alive_seq.shape[-1])
                attention = (
                    alive_attn.view(
                        alive_attn.shape[0], -1, beam_size, alive_attn.shape[-1])
                    if alive_attn is not None else None)
                for i in range(is_finished.shape[0]):
                    b = batch_offset[i]
                    if(end_condition[i]):
                        is_finished[i].fill_(1)
                    finished_hyp = is_finished[i].nonzero().view(-1)
                    # Store finished hypotheses for this batch.
                    for j in finished_hyp:
                        hypotheses[b].append((
                            topk_scores[i, j],
                            predictions[i, j, 1:],  # Ignore start_token.
                            attention[:, i, j, :memory_length[i]]
                            if attention is not None else None))
                    # If the batch reached the end, save the n_best hypotheses.
                    if end_condition[i]:
                        best_hyp = sorted(
                            hypotheses[b], key=lambda x: x[0], reverse=True)
                        for n, (score, pred, attn) in enumerate(best_hyp):
                            if n >= n_best:
                                break
                            results["scores"][b].append(score)
                            results["predictions"][b].append(pred)
                            results["attention"][b].append(
                                attn if attn is not None else [])
                non_finished = end_condition.eq(0).nonzero().view(-1)
                # If all sentences are translated, no need to go further.
                if len(non_finished) == 0:
                    break
                # Remove finished batches for the next step.
                topk_log_probs = topk_log_probs.index_select(0, non_finished)
                batch_index = batch_index.index_select(0, non_finished)
                batch_offset = batch_offset.index_select(0, non_finished)
                alive_seq = predictions.index_select(0, non_finished) \
                                       .view(-1, alive_seq.size(-1))
                if alive_attn is not None:
                    alive_attn = attention.index_select(1, non_finished) \
                                          .view(alive_attn.size(0),
                                                -1, alive_attn.size(-1))

            # Reorder states.
            select_indices = batch_index.view(-1)
            memory_bank = memory_bank.index_select(1, select_indices)
            memory_length = memory_length.index_select(0, select_indices)
            dec_states.map_batch_fn(
                lambda state, dim: state.index_select(dim, select_indices))

        sys.exit(-1)
        #print('results["predictions"]',results["predictions"])

        return results

    def _translate_batch(self,batch):
        # (0) Prep each of the components of the search.
        # And helper method for reducing verbosity.
        beam_size = self.beam_size
        batch_size = batch['source'].shape[1]
        
        vocab = self.vocab['target']

        # Define a list of tokens to exclude from ngram-blocking
        # exclusion_list = ["<t>", "</t>", "."]
        exclusion_token = set([vocab[t]
                                for t in self.ignore_when_blocking])

        beam = [Beam.beam(beam_size,n_best=self.n_best,
                            cuda=self.cuda,global_scorer=self.global_scorer,
                            pad=vocab[Constant.PAD_token],
                            eos=vocab[Constant.EOS_token],
                            bos=vocab[Constant.SOS_token],
                            min_length=self.min_length,
                            stepwise_penalty=self.stepwise_penalty,
                            block_ngram_repeat=self.block_ngram_repeat,
                            exclusion_tokens=exclusion_token
                            ) for _ in range(batch_size)]
        def var(a):
            return torch.tensor(a,requires_grad = False)
        
        def rvar(a):
            return var(a.repeat(1,beam_size,1))
        
        def bottle(m):
            return m.view(batch_size * beam_size,-1)
        
        def unbottle(m):
            return m.view(beam_size,batch_size,-1)
        
        enc_state,memory_bank,_ = self.model.encoder(batch['source'],batch['source_len'])
        dec_state = self.model.decoder.init_decoder_state(
                        batch['source'],memory_bank,enc_state)

        if(isinstance(memory_bank,tuple)):
            memory_bank = tuple(rvar(x.data) for x in memory_bank)
        else:
            memory_bank = rvar(memory_bank.data)

        memory_length = batch['source_len'].repeat(beam_size)
        dec_state.repeat_beam_size_times(beam_size)

        for i in range(self.max_length):
            if(all((b.done() for b in beam))):
                break

            inp = var(torch.stack([b.get_current_state() for b in beam])
                    .t().contiguous().view(1,-1))

            inp = inp.unsqueeze(2)

            dec_out, dec_state, attn = self.model.decoder(
                inp,memory_bank,dec_state,
                memory_length=memory_length,step=i
            )

            dec_out = dec_out.squeeze(0)

            #we dont use copy attention here
            if(not self.copy_attn):
                out = F.log_softmax(dec_out).data

                out = unbottle(out)
                beam_attn = unbottle(attn["std"])

            for j,b in enumerate(beam):
                b.advance(out[:,j],
                        beam_attn.data[:,j,:memory_length[j]])
                dec_state.beam_update(j,b.get_current_origin(),beam_size)


        ret = self._from_beam(beam)
        #here means that it has the reference answer
        ret["gold_score"] = [0] * batch_size
        if("target" in batch.__dict__):
            ret['gold_score'] = self._run_target(batch)
        ret['batch'] = batch

        return ret

    def _from_beam(self,beam):
        ret = {"predictions":[],
                "scores":[],
                "attention":[]}
        
        for b in beam:
            n_best = self.n_best
            scores,ks = b.sort_finished(minium=n_best)
            hyps, attn = [], []
            for i ,(times,k) in enumerate(ks[:n_best]):
                hyp, att = b.get_hyp(times,k)
                hyps.append(hyp)
                attn.append(att)
            ret['predictions'].append(hyps)
            ret['scores'].append(scores)
            ret['attention'].append(attn)
        return ret

    def _run_target(self,batch):
        #i think it might not be use, we pass it first
        pass
        return None

        
    def _report_score(self, name, score_total, words_total):
        if(words_total == 0):
            msg = "%s no words predicted" % (name,)
        else:
            temp = score_total / words_total
            msg = ("%s AVG SCORE: %.4f, %s PPl: %.4f" %(name,
            temp , name,math.exp(-temp)))
        return msg

    def _report_bleu(self,target_path):
        import subprocess
        base_dir = os.path.abspath(__file__ + "/../../..")
        
        self.out_file.seek(0)

        res = subprocess.check_output("perl %s/tools/multi-bleu.perl %s"
                                        % (base_dir, target_path),
                                        stdin=self.out_file,
                                        shell=True).decode("utf-8")

        msg = ">> " + res.strip()
        return msg

    def _report_rouge(self, tgt_path):
        import subprocess
        path = os.path.split(os.path.realpath(__file__))[0]
        res = subprocess.check_output(
            "python %s/tools/test_rouge.py -r %s -c STDIN"
            % (path, tgt_path),
            shell=True,
            stdin=self.out_file).decode("utf-8")
        msg = res.strip()
        return msg