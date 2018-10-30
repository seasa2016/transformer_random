import torch
import six
import unicodedata
import sys

ALPHANUMERIC_CHAR_SET = set(
    six.unichr(i) for i in range(sys.maxunicode)
    if (unicodedata.category(six.unichr(i)).startswith("L") or
        unicodedata.category(six.unichr(i)).startswith("N")))


class TranslationBuilder(object):
    """
    Build a word-based translation from the batch output
    of translator and the underlying dictionaries.

    Replacement based on "Addressing the Rare Word
    Problem in Neural Machine Translation" :cite:`Luong2015b`

    Args:
       data (DataSet):
       fields (dict of Fields): data fields
       n_best (int): number of translations produced
       replace_unk (bool): replace unknown words using attention
       has_tgt (bool): will the batch have gold targets
    """

    def __init__(self,n_best=1,replace_unk=False,has_target=False):
        self.n_best = n_best
        self.replace_unk = replace_unk
        self.has_target = has_target
        
        self.vocab = {}
        for t in ['source','target','tag']:
            self.vocab[t] = []
            with open('./{0}/subword.{1}'.format(opt.data,t)) as f:
                for word in f:
                    if(t=='target'):
                        word = word.strip()+'_'
                    else:
                        word = word.strip()[1:-1].replace('_',' ')
                        word = word.replace('\\u','_')
                    
                    self.vocab[t].append(word)

    def _build_sentence(self,pred,ttype=None):
        """
            convert the data into sentence
        """
        if(ttype is None):
            raise ValueError("you should choose a source type to convert")

        vocab = self.vocab[ttype]
        temp = []
        
        if(ttype=='source'):
            temp = [ self.vocab['tag'][pred[0]] ]

        for _ in pred[1:]:
            try:
                temp.append(vocab[_])
            except KeyError as e:
                raise KeyError('the word {0} is unsee '.format(e.args[0]))
        #print(temp)
        if(ttype=='source'):
            target = ''+temp[0]+' '
            for i in range(1,len(temp)):
                try:
                    if((temp[i-1][0] in ALPHANUMERIC_CHAR_SET ) != (temp[i][0] in ALPHANUMERIC_CHAR_SET )):
                        target = target[:-1]    
                    target += temp[i]
                except IndexError:
                    raise ValueError("why error with no word")
        else:
            target = ' '.join(temp)

        return target

    def from_batch(self,translation_batch,num):
        batch = translation_batch["batch"]

        assert(len(translation_batch["gold_score"]) == len(translation_batch["predictions"]))

        batch_size = batch['source'].shape[1]
        indices_temp = torch.tensor([num+i for i in range(batch_size)]).cuda()

        preds, pred_score, attn, gold_score, indices = list(zip(
            *sorted(zip(translation_batch["predictions"],
                        translation_batch["scores"],
                        translation_batch["attention"],
                        translation_batch["gold_score"],
                        indices_temp),
                    key=lambda x: x[-1])))
        # Sorting
        inds, perm = torch.sort(indices_temp)
        
        src = batch['source'].index_select(1, perm)
        
        if('target' in batch):
            tgt = batch['target'].index_select(1, perm)
        else:
            tgt = None

        translations = []
        for b in range(batch_size):
            pred_sents = [self._build_sentence(
                preds[b][n],ttype='target')
                for n in range(self.n_best)]
            
            gold_sent = None
            #this works only with target given for checking performance
            if('target' in  batch):
                gold_sent = self._build_sentence(tgt[1:, b] if tgt is not None else None,ttype='target')

            src_sent = self._build_sentence(src[:, b] if src is not None else None,ttype='source')
            translation = Translation(  src_sent,
                                        pred_sents,
                                        attn[b], pred_score[b], gold_sent,
                                        gold_score[b])
            translations.append(translation)

        return translations

class Translation(object):
    """
    Container for a translated sentence.

    Attributes:
        src (`LongTensor`): src word ids
        src_raw ([str]): raw src words

        pred_sents ([[str]]): words from the n-best translations
        pred_scores ([[float]]): log-probs of n-best translations
        attns ([`FloatTensor`]) : attention dist for each translation
        gold_sent ([str]): words from gold translation
        gold_score ([float]): log-prob of gold translation

    """

    def __init__(self,src,pred_sent,attn,
                pred_scores,target_sent,target_scores):
        
        self.src = src
        self.attn = attn
        #this is for the prob of construction
        self.pred_scores = pred_scores
        self.pred_sent = pred_sent
        
        self.target_sent =target_sent
        self.target_scores = target_scores

    def log(self, sent_number):
        """
        Log translation.
        """

        output = '\nSENT {}: {}\n'.format(sent_number, self.src)

        best_pred = self.pred_sent[0]
        best_score = self.pred_scores[0]
        pred_sent = best_pred
        output += 'PRED {}: {}\n'.format(sent_number, pred_sent)
        output += "PRED SCORE: {:.4f}\n".format(best_score)

        if self.target_sent is not None:
            tgt_sent = self.target_sent
            output += 'GOLD {}: {}\n'.format(sent_number, tgt_sent)
            output += ("GOLD SCORE: {:.4f}\n".format(self.target_scores))
        if len(self.pred_sent) > 1:
            output += '\nBEST HYP:\n'
            for score, sent in zip(self.pred_scores, self.pred_sent):
                output += "[{:.4f}] {}\n".format(score, sent)

        return output
