from game import TrAIlCrossers
from A2C import A2CTrainer

import cProfile
import pstats
from io import StringIO
import tensorflow as tf

pr = cProfile.Profile()
pr.enable()
s = StringIO()

PLAYER_COUNT = 2

# tf.compat.v1.disable_eager_execution()
with tf.device('/GPU:0'):
    A2CTrainer(TrAIlCrossers(PLAYER_COUNT), PLAYER_COUNT, "TrAIlCrossers-A2C-Basic", 50).run(5000)

pr.disable()
ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
ps.print_stats()
print(s.getvalue())
