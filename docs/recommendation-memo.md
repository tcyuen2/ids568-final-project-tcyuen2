# Recommendation Memo — Iris Classifier Model A/B Test

I recommend ship variant B. This is because there is a ~3% accuracy gap that was seen offline and din't hold up. In the A/B test, both models performed about the same but B was slightly better than A (0.981 vs 0.977). Ship variant B was also cheaper and faster to run so I think it would be the better choice.

From our results, using the offline, small sample, we got:
   A = 0.967 and B = 0.933.

In this case A looks better, but on the A/B test and realistic scale, our results were:
   A=0.977 and B=0.981

The offline test only had 30 samples, which made it easy for random noise to create a fake gap. When we has thousands of requests in our A/B test, the fake gap seems to disappear and B becomes the better option.

Even if we ignore the accuracies, B is more efficient because:
- there are 2x fewer treas which mean faster ingerence
- trees are shallower, so we use a smaller model
- there is lower latency

The accuracies of A and B are very similar, but B has a much lower cost than A.

Summary: Variant B perform just a well as A (slightly better), but it costs less to run and scales better.

Choice: *Ship variant B*