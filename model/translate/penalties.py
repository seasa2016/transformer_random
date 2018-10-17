import torch

class PenaltyBuilder(object):
    """
    Returns the Length and lengtherage Penalty function for Beam Search.

    Args:
        length_pen (str): option name of length pen
        length_pen (str): option name of length pen
    """
    def __init__(self, length_pen, cov_pen):
        self.length_pen = length_pen
        self.cov_pen = cov_pen

    def coverage_penalty(self):
        if(self.length_pen == "wu"):
            return self.coverage_wu
        elif(self.length_pen =="summary"):
            return self.coverage_summary
        else:
            return self.coverage_none
    
    def length_penalty(self):
        if(self.length_pen == "wu"):
            return self.length_wu
        elif(self.length_pen =="avg"):
            return self.length_average
        else:
            return self.length_none
    
    def coverage_wu(self,beam,cov,beta=0.):
        """
        NMT coverage re-ranking score from
        "Google's Neural Machine Translation System" :cite:`wu2016google`.
        """
        penalty = -torch.min( cov,cov.clone().fill_(1.0).log().sum(1) )
        return beta*penalty
    def coverage_summary(self,beam,cov,beta=0.):
        """
        their summary penalty
        """
        penalty = torch.max(cov, cov.clone().fill_(1.0)).sum(1)
        penalty -= cov.size(1)
        return beta * penalty
    def coverage_none(self, beam, cov, beta=0.):
        """
        returns zero as penalty
        """
        return beam.scores.clone().fill_(0.0)

    def length_wu(self, beam, logprobs, alpha=0.):
        """
        NMT length re-ranking score from
        "Google's Neural Machine Translation System" :cite:`wu2016google`.
        """

        modifier = (((5 + len(beam.next_ys)) ** alpha) /
                    ((5 + 1) ** alpha))
        return (logprobs / modifier)

    def length_average(self, beam, logprobs, alpha=0.):
        """
        Returns the average probability of tokens in a sequence.
        """
        return logprobs / len(beam.next_ys)

    def length_none(self, beam, logprobs, alpha=0., beta=0.):
        """
        Returns unmodified scores.
        """
        return logprobs


