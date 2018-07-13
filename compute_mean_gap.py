
import sys
from os.path import join

from tensorflow import app
from tensorflow import flags

FLAGS = flags.FLAGS

flags.DEFINE_string('log_dir', None, 
                    """Directory with log file to evaluate the GAP.""")
flags.DEFINE_string('steps', None, 
                    """Evalutate mean GAP between steps '2000,3000'.""")


def main(_):

  if FLAGS.steps is None:
    print('--steps needs to be defined')
    sys.exit(0)
  if FLAGS.log_dir is None:
    print('--log_dir needs to be defined')
    sys.exit(0)

  with open(join(FLAGS.log_dir, 'log_master_0.logs')) as f:
    gap_lines = [line for line in f.readlines() if 'GAP' in line]

  step_start, start_end = list(map(int, FLAGS.steps.split(',')))
  gap_accumulate, count = 0, 0
  for line in gap_lines:
    line = line.split(' ')
    step = int(line[2])
    gap = float(line[-1])
    if step >= start_end:
      break
    if step >= step_start:
      gap_accumulate += gap
      count += 1

  mean_gap = gap_accumulate / count
  print("mean train GAP between {} and {} : {:.3f}".format(
    step_start, start_end, mean_gap))


if __name__ == '__main__':
  app.run()
