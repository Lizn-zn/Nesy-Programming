'''Nonograms solver.

Nonograms is a perfect-information logic puzzle with simple rules.
There is a free online implementation at, for instance,
https://www.puzzle-nonograms.com/.

The rules are as follows: You have a grid of squares, which must be
either filled in black or left white. Beside each row of the grid are
listed the lengths of the runs of black squares on that row. Above
each column are listed the lengths of the runs of black squares in
that column.  The aim is to find all black squares.

This module encodes that puzzle definition as a finite-domain
constraint satisfaction problem and solves it with Z3
(https://github.com/Z3Prover/z3).  In my limited testing, Z3 finds
solutions to puzzles up to 25x25 in a matter of seconds.

For this to work, run it in a virtualenv that has the Z3 python
bindings installed.

'''
import sys
import numpy as np
import z3

# Several example puzzles below.

# The rows are encoded from the top down, with each row's entries from
# left to right.
rows_5 = [
  [1],
  [1, 1],
  [4],
  [3],
  [3]]

# The columns are encoded from left to right, with each column's
# entries from the top down.
columns_5 = [
  [3, 1],
  [3],
  [3],
  [2],
  [1]]

rows_10 = [
  [2, 4],
  [8],
  [4, 1],
  [1, 1, 4],
  [3, 3],
  [3, 4],
  [3, 3],
  [1, 2],
  [2],
  [1]]

columns_10 = [
  [8],
  [3, 3],
  [6],
  [2],
  [1, 1, 2],
  [2, 1, 3],
  [2, 4],
  [7],
  [1, 3],
  [1]]

rows_10b = [
  [1],
  [2],
  [1, 2],
  [7],
  [1, 6],
  [1, 1, 4],
  [5],
  [6],
  [6],
  [7]]

columns_10b = [
  [3],
  [1],
  [3],
  [3, 1],
  [3, 3],
  [7],
  [7],
  [6],
  [2, 4],
  [3, 4]]

rows_15 = [
  [3, 3],
  [3],
  [4, 4],
  [1, 1, 1, 1],
  [1, 3, 4],
  [5, 1],
  [2, 3, 1],
  [2, 1],
  [2, 1, 2, 1],
  [2, 5],
  [4, 6, 2],
  [2, 1, 1, 4, 1],
  [8, 4],
  [12],
  [4, 3, 4]]

columns_15 = [
  [4, 5],
  [3, 6],
  [3, 3, 3],
  [3, 1, 5],
  [4, 2],
  [1, 5],
  [3, 3, 3],
  [1, 1, 2, 6],
  [1, 3, 3, 1],
  [1, 1, 1, 7],
  [8],
  [1, 3],
  [3, 1, 1],
  [3, 1],
  [3, 1, 1, 2]]

rows_20 = [
  [2, 5],
  [1, 3, 4],
  [1, 2, 1, 1],
  [2, 3],
  [2, 3, 3],
  [2, 7, 1],
  [2, 6, 2],
  [2, 5, 3],
  [7, 2, 4],
  [1, 2, 7, 4],
  [4, 6, 3],
  [4, 6, 3],
  [3, 7, 1],
  [10, 1],
  [1, 3, 2, 3, 1],
  [5, 2, 2, 2],
  [8, 4, 2],
  [11, 2],
  [2, 1, 3, 1],
  [2, 3, 2]]

columns_20 = [
  [4, 4, 6],
  [4, 3, 5],
  [3, 3],
  [1, 4, 4],
  [6, 2, 5],
  [6, 2, 3],
  [4, 1, 2],
  [13],
  [13, 1, 1],
  [2, 11, 3],
  [6, 5, 4],
  [2, 6, 1, 1],
  [9],
  [3, 2, 5, 1],
  [2, 2],
  [3, 1],
  [2, 3],
  [1, 5],
  [4, 1, 3],
  [5, 7]]

rows_25 = [
  [3, 2, 1],
  [2, 1, 1, 2],
  [4],
  [1, 2, 4],
  [2, 1, 1, 1, 3, 3],
  [3, 4, 3, 8],
  [1, 14],
  [1, 4, 8],
  [2, 3, 4],
  [5, 6, 3],
  [5, 6, 4],
  [6, 11, 3],
  [2, 1, 5, 2, 4],
  [2, 1, 4, 3, 2],
  [2, 1, 1, 3, 1, 1],
  [2, 3, 1, 3, 1],
  [1, 5, 3, 3, 1],
  [9, 4, 2],
  [5, 7, 3, 2],
  [4, 5, 2, 1],
  [5, 1, 1, 2, 3],
  [4, 1, 6, 1],
  [8, 1, 1, 1, 1],
  [18, 1],
  [10, 8]]

columns_25 = [
  [11, 5],
  [2, 9, 1, 5],
  [3, 3, 7],
  [3, 7],
  [1, 3, 6, 3],
  [1, 1, 1, 5, 3],
  [2, 2, 4, 5],
  [2, 1, 5, 2, 3],
  [2, 3, 5, 2],
  [8, 2, 2],
  [15, 1],
  [20],
  [1, 11, 1, 1],
  [1, 2, 3, 4, 2],
  [1, 4, 2, 4, 2],
  [2, 11, 1, 2],
  [1, 12, 1, 2],
  [8, 1, 1, 1, 2],
  [3, 7, 1],
  [3, 5, 1],
  [4, 3, 1, 1],
  [7, 1, 1],
  [3, 2, 1],
  [3, 3, 2],
  [3, 4, 4]]

def nonograms(rows, columns):
  '''Encode the given Nonograms puzzle as a collection of Z3 constraints.

  Return a 2-tuple of the Z3 Solver object representing the puzzle,
  and the Z3 Bool objects representing whether each square in the grid
  is black.  The squares are indexed row-major.

  This function just encodes the problem; to solve it, invoke
  `.check()` on the returned Solver.

  '''
  width = len(columns)
  height = len(rows)
  s = z3.Solver()
  s.set(unsat_core=True)
  squares = [[z3.Bool('sq %d %d' % (i, j)) for i in range(width)] for j in range(height)]

  # Horizontal block constraints
  horiz_block_positions = {}
  horiz_block_lengths = {}
  for (j, row) in enumerate(rows):
    for (k, block_len) in enumerate(row):
      left_edge = z3.Int('horiz block %d %d' % (k, j))
      horiz_block_positions[(k, j)] = left_edge
      horiz_block_lengths[(k, j)] = block_len
      s.assert_and_track(left_edge >= 0, 'left edge positive %d %d' % (j, k))
      s.assert_and_track(left_edge <= width - block_len, 'left edge in bounds %d %d' % (j, k))
      if k > 0:
        prev_end = horiz_block_positions[(k-1, j)] + horiz_block_lengths[(k-1, j)]
        if block_len > 0:
          s.assert_and_track(left_edge >= prev_end + 1, 'horiz block separation %d %d' % (j, k))
      for i in range(width):
        # If the square is in the block, it must be on
        s.assert_and_track(z3.Implies(z3.And(left_edge <= i, left_edge + block_len > i), squares[j][i]),
                           'squares on horiz %d %d %d' % (j, k, i))
        # There are three ways the square might not be in any block in
        # the row, in which case it must be off
        if k > 0:
          s.assert_and_track(z3.Implies(z3.And(left_edge > i, prev_end <= i), z3.Not(squares[j][i])),
                             'gap squares off horiz %d %d %d' % (j, k, i))
        else:
          s.assert_and_track(z3.Implies(left_edge > i, z3.Not(squares[j][i])),
                             'early squares off horiz %d %d %d' % (j, k, i))
        if k == len(row) - 1:
          s.assert_and_track(z3.Implies(left_edge + block_len <= i, z3.Not(squares[j][i])),
                             'late squares off horiz %d %d %d' % (j, k, i))

  # Vertical block positioning and length constraints
  vert_block_positions = {}
  vert_block_lengths = {}
  for (i, col) in enumerate(columns):
    for (k, block_len) in enumerate(col):
      top_edge = z3.Int('vert block %d %d' % (i, k))
      vert_block_positions[(i, k)] = top_edge
      vert_block_lengths[(i, k)] = block_len
      s.assert_and_track(top_edge >= 0, 'top edge positive %d %d' % (i, k))
      s.assert_and_track(top_edge <= height - block_len, 'top edge in bounds %d %d' % (i, k))
      if k > 0:
        prev_end = vert_block_positions[(i, k-1)] + vert_block_lengths[(i, k-1)]
        if block_len > 0:
          s.assert_and_track(top_edge >= prev_end + 1, 'vert block separation %d %d' % (i, k))
      for j in range(height):
        # If the square is in the block, it must be on
        s.assert_and_track(z3.Implies(z3.And(top_edge <= j, top_edge + block_len > j), squares[j][i]),
                           'squares on vert %d %d %d' % (i, k, j))
        # There are three ways the square might not be in any block in
        # the column, in which case it must be off
        if k > 0:
          s.assert_and_track(z3.Implies(z3.And(top_edge > j, prev_end <= j), z3.Not(squares[j][i])),
                             'gap squares off vert %d %d %d' % (i, k, j))
        else:
          s.assert_and_track(z3.Implies(top_edge > j, z3.Not(squares[j][i])),
                             'early squares off vert %d %d %d' % (i, k, j))
        if k == len(col) - 1:
          s.assert_and_track(z3.Implies(top_edge + block_len <= j, z3.Not(squares[j][i])),
                             'late squares off vert %d %d %d' % (i, k, j))
  return s, squares

def visualize(model, squares, rows, columns):
  '''Print ASCII art describing the solution found in the given model.'''
  for j in range(len(rows)):
    for i in range(len(columns)):
      char = 'X' if model[squares[j][i]] else '.'
      sys.stdout.write(char)
    print('')

def solve(rows, columns):
  '''Fully solve the given Nonograms puzzle, and print the solution as
  ASCII art if it exists.'''
  (s, squares) = nonograms(rows, columns)
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
  solve(rows_25, columns_25)