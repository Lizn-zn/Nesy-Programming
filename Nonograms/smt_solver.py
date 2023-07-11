import sys
from z3 import *
import numpy as np

# Several example puzzles below.

# The rows are encoded from the top down, with each row's entries from
# left to right.
rows_5 = [
  [2, 0],
  [1, 2],
  [4, 0],
  [1, 1],
  [1, 1]]

# The columns are encoded from left to right, with each column's
# entries from the top down.
columns_5 = [
  [2, 0],
  [3, 0],
  [1, 0],
  [5, 0],
  [2, 0]]


def nonograms(rows, columns, W, b):
  '''Encode the given Nonograms puzzle as a collection of Z3 constraints.

  Return a 2-tuple of the Z3 Solver object representing the puzzle,
  and the Z3 Bool objects representing whether each square in the grid
  is black.  The squares are indexed row-major.

  This function just encodes the problem; to solve it, invoke
  `.check()` on the returned Solver.

  '''
  width = len(columns)
  height = len(rows)
  # s = Optimize()
  s = Solver()
  # s.set(unsat_core=True)
  squares = [[Bool('sq %d %d' % (i, j)) for i in range(width)] for j in range(height)]

  # Horizontal block constraints
  horiz_block_positions = {}
  horiz_block_lengths = {}
  for (j, row) in enumerate(rows):
    X_seq = []
    Z_seq = []
    for (k, block_len) in enumerate(row):
      left_edge = Int('horiz block %d %d' % (k, j))
      X = Bool('row_aux_x %d %d' % (j, k))
      Z = Bool('row_aux_z %d %d' % (j, k)) 
      X_seq.append(X); Z_seq.append(Z)

      horiz_block_positions[(k, j)] = left_edge
      horiz_block_lengths[(k, j)] = block_len
      s.assert_and_track(left_edge >= 0, 'left edge positive %d %d' % (j, k))
      s.assert_and_track(left_edge <= width - block_len, 'left edge in bounds %d %d' % (j, k))
      ########
      if k == 0:
        s.assert_and_track(Z == (left_edge > 0), 'row separation positive %d %d' % (j, k))
      else:
        prev_end = horiz_block_positions[(k-1, j)] + horiz_block_lengths[(k-1, j)]
        s.assert_and_track(Z == (left_edge >= prev_end + 1), 'row separation positive %d %d' % (j, k))
      if k == len(row) - 1:
        Z = Bool('row_aux_z %d %d ' % (j, k+1)) 
        Z_seq.append(Z)
        s.assert_and_track(Z == (width >= left_edge + block_len + 1), 'row separation positive %d %d' % (j, k+1))
      #######
      s.assert_and_track(X == (block_len >= 1), 'row block exists %d %d' % (j, k))

      for i in range(width):
        # If the square is in the block, it must be on
        s.assert_and_track(z3.Implies(And(left_edge <= i, left_edge + block_len > i), squares[j][i]),
                           'squares on horiz %d %d %d' % (j, k, i))
        # There are three ways the square might not be in any block in
        # the row, in which case it must be off
        if k > 0:
          s.assert_and_track(Implies(And(left_edge > i, prev_end <= i), Not(squares[j][i])),
                             'gap squares off horiz %d %d %d' % (j, k, i))
        else:
          s.assert_and_track(Implies(left_edge > i, Not(squares[j][i])),
                             'early squares off horiz %d %d %d' % (j, k, i))
        if k == len(row) - 1:
          s.assert_and_track(Implies(left_edge + block_len <= i, Not(squares[j][i])),
                             'late squares off horiz %d %d %d' % (j, k, i))

    x_range = len(X_seq)
    z_range = len(Z_seq)
    for Wtmp, btmp in zip(W, b):
      eqn = []
      for t in btmp:
        eqn.append(PbEq([(Not(X_seq[i]), Wtmp[i]) for i in range(x_range)] + [(Z_seq[i], Wtmp[i+x_range]) for i in range(z_range)], t))
      # s.add_soft(Or(eqn), 1)
      s.add(Or(eqn))

  # Vertical block positioning and length constraints
  vert_block_positions = {}
  vert_block_lengths = {}
  for (i, col) in enumerate(columns):
    X_seq = []
    Z_seq = []
    for (k, block_len) in enumerate(col):
      top_edge = Int('vert block %d %d' % (i, k))
      X = Bool('col_aux_x %d %d' % (i, k))
      Z = Bool('col_aux_z %d %d ' % (i, k)) 
      X_seq.append(X); Z_seq.append(Z)

      vert_block_positions[(i, k)] = top_edge
      vert_block_lengths[(i, k)] = block_len
      s.assert_and_track(top_edge >= 0, 'top edge positive %d %d' % (i, k))
      s.assert_and_track(top_edge <= height - block_len, 'top edge in bounds %d %d' % (i, k))
      ########
      if k == 0:
        s.assert_and_track(Z == (top_edge > 0), 'col separation positive %d %d' % (i, k))
      else:
        prev_end = vert_block_positions[(i, k-1)] + vert_block_lengths[(i, k-1)]
        s.assert_and_track(Z == (top_edge >= prev_end + 1), 'col separation positive %d %d' % (i, k))
      if k == height - 1:
        Z = Bool('col_aux0 %d ' % (k+1)) 
        Z_seq.append(Z)
        s.assert_and_track(Z == (height >= top_edge + block_len + 1), 'col separation positive %d %d' % (i, k))
    #   ########
      s.assert_and_track(X == (block_len >= 1), 'col block exists %d %d' % (i, k))

      for j in range(height):
        # If the square is in the block, it must be on
        s.assert_and_track(Implies(And(top_edge <= j, top_edge + block_len > j), squares[j][i]),
                           'squares on vert %d %d %d' % (i, k, j))
        # There are three ways the square might not be in any block in
        # the column, in which case it must be off
        if k > 0:
          s.assert_and_track(Implies(And(top_edge > j, prev_end <= j), Not(squares[j][i])),
                             'gap squares off vert %d %d %d' % (i, k, j))
        else:
          s.assert_and_track(Implies(top_edge > j, Not(squares[j][i])),
                             'early squares off vert %d %d %d' % (i, k, j))
        if k == len(col) - 1:
          s.assert_and_track(Implies(top_edge + block_len <= j, Not(squares[j][i])),
                             'late squares off vert %d %d %d' % (i, k, j))

    x_range = len(X_seq)
    z_range = len(Z_seq)
    for Wtmp, btmp in zip(W, b):
      eqn = []
      for t in btmp:
        eqn.append(PbEq([(Not(X_seq[i]), Wtmp[i]) for i in range(x_range)] + [(Z_seq[i], Wtmp[i+x_range]) for i in range(z_range)], t))
      # s.add_soft(Or(eqn), 1)
      s.add(Or(eqn))

  return s, squares

def visualize(model, squares, rows, columns):
  '''Print ASCII art describing the solution found in the given model.'''
  for j in range(len(rows)):
    for i in range(len(columns)):
      char = 'X' if model[squares[j][i]] else '.'
      sys.stdout.write(char)
    print('')

def solve(rows, columns, W, b):
  '''Fully solve the given Nonograms puzzle, and print the solution as
  ASCII art if it exists.'''
  (s, squares) = nonograms(rows, columns, W, b)
  res = s.check()
  if res == z3.sat:
    # print("Solution")
    m = s.model()
    sol = np.zeros((len(rows), len(columns)))
    for j in range(len(rows)):
      for i in range(len(columns)):
        if m[squares[j][i]]:
          sol[j,i] = 1
    return sol
    # visualize(m, squares, rows, columns)
  else:
    print("Unsolvable because")
    print(s.unsat_core())


if __name__ == '__main__':
  solve(rows_5, columns_5, W=[[1,1,0,1,0]], b=[[3,1]])