"""Utils"""

import os

DFLT_DIRPATH = os.path.expanduser('~')

from itertools import islice
from typing import Optional, Iterable


def fixed_step_chunker(
    it: Iterable,
    chk_size: int,
    chk_step: Optional[int] = None,
    start_at: Optional[int] = None,
    stop_at: Optional[int] = None,
    return_tail: bool = False,
):
    """
      a function to get (an iterator of) segments (bt, tt) of chunks from an iterator (or list)
      of the for [it_1, it_2...], given a chk_size, chk_step, and a start_at and a stop_at.
      The start_at, stop_at act like slices indices for a list: start_at is included and stop_at is excluded

      Args:
          it: iterator of elements of any type
          chk_size: length of the chunks
          chk_step: step between chunks
          start_at: index of the first term of the iterator at which we begin building the chunks (inclusive)
          stop_at: index of the last term from the iterator included in the chunks
          return_tail: if set to false, only the chunks with max element with index less than stop_at are yielded
              if set to true, any chunks with minimum index value no more than stop_at are returned but they contain
              term with index no more than stop_at
      
      Returns: 
          an iterator of the chunks

      1) If stop_at is not None and return_tail is False:
         will return all full chunks with maximum element index less than stop_at
         or until the iterator is exhausted. Only full chunks are returned here.

      2) If stop_at is not None and return_tail is True:
         will return all full chunks as above along with possibly cut off chunks
         containing one term whose index is stop_at-1 or one (last) term which is the
         last element of it

      3) If stop_at is None and return_tail is False:
         will return all full chunks with maximum element index less or equal to the last
         element of it

      4) If stop_at is None and return_tail is True:
         will return all full chunks with maximum element index less or equal to the last
         element of it plus cut off chunks whose maximum term index is the last term of it

      Examples:

        >>> list(fixed_step_chunker([1,2,3,4,5,6,7], chk_size=3, chk_step=2))
        [[1, 2, 3], [3, 4, 5], [5, 6, 7]]

        Testing chk_step < chk_size with return_tail=TRUE, stop and start_at PRESENT
        and stop_at SMALLER than the largest index of it

        >>> f = lambda it: fixed_step_chunker(it, chk_size=3, chk_step=1, start_at=2, stop_at=5, return_tail=True)
        >>> it = list(range(1, 17, 1))
        >>> A = list(f(it)); B = list(f(iter(it)));  # trying the function on it (a list) and iter(it) (and iterator)
        >>> assert A == B  # it and iter(it) should give the same thing!
        >>> A  # and that thing is:
        [[3, 4, 5], [4, 5], [5]]

        Testing chk_step < chk_size with return_tail=FALSE, stop and start_at PRESENT
        and stop_at SMALLER than the largest index of it

        >>> f = lambda it: fixed_step_chunker(it, chk_size=3, chk_step=1, start_at=2, stop_at=5, return_tail=False)
        >>> it = list(range(1, 17, 1))
        >>> A = list(f(it)); B = list(f(iter(it)));  # trying the function on it (a list) and iter(it) (and iterator)
        >>> assert A == B  # it and iter(it) should give the same thing!
        >>> A  # and that thing is:
        [[3, 4, 5]]

        Testing chk_step < chk_size with return_tail=TRUE, stop and start_at PRESENT
        and stop_at LARGER than the largest index of it

        >>> f = lambda it: fixed_step_chunker(it, chk_size=3, chk_step=1, start_at=1, stop_at=20, return_tail=True)
        >>> it = list(range(1, 17, 1))
        >>> A = list(f(it)); B = list(f(iter(it)));  # trying the function on it (a list) and iter(it) (and iterator)
        >>> assert A == B  # it and iter(it) should give the same thing!
        >>> A  # and that thing is:
        [[2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8], [7, 8, 9], [8, 9, 10], [9, 10, 11], [10, 11, 12], [11, 12, 13], [12, 13, 14], [13, 14, 15], [14, 15, 16], [15, 16], [16]]

        Testing chk_step < chk_size with return_tail=FALSE, stop and start_at PRESENT
        and stop_at LARGER than the largest index of it

        >>> f = lambda it: fixed_step_chunker(it, chk_size=3, chk_step=1, start_at=1, stop_at=20, return_tail=False)
        >>> it = list(range(1, 17, 1))
        >>> A = list(f(it)); B = list(f(iter(it)));  # trying the function on it (a list) and iter(it) (and iterator)
        >>> assert A == B  # it and iter(it) should give the same thing!
        >>> A  # and that thing is:
        [[2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8], [7, 8, 9], [8, 9, 10], [9, 10, 11], [10, 11, 12], [11, 12, 13], [12, 13, 14], [13, 14, 15], [14, 15, 16]]

        Testing chk_step = chk_size with return_tail=TRUE, stop and start_at PRESENT
        and stop_at SMALLER than the largest index of it

        >>> f = lambda it: fixed_step_chunker(it, chk_size=3, chk_step=3, start_at=1, stop_at=7, return_tail=True)
        >>> it = list(range(1, 17, 1))
        >>> A = list(f(it)); B = list(f(iter(it)));  # trying the function on it (a list) and iter(it) (and iterator)
        >>> assert A == B  # it and iter(it) should give the same thing!
        >>> A  # and that thing is:
        [[2, 3, 4], [5, 6, 7]]

        Testing chk_size > len(it) with return_tail=False, no stop_at or start_at

        >>> f = lambda it: fixed_step_chunker(it, chk_size=30, chk_step=3, start_at=None, stop_at=None, return_tail=False)
        >>> it = list(range(1, 17, 1))
        >>> A = list(f(it)); B = list(f(iter(it)));  # trying the function on it (a list) and iter(it) (and iterator)
        >>> assert A == B  # it and iter(it) should give the same thing!
        >>> A  # and that thing is:
        []

        Testing chk_size > len(it) with return_tail=True, no stop_at or start_at

        >>> f = lambda it: fixed_step_chunker(it, chk_size=30, chk_step=3, start_at=None, stop_at=None, return_tail=True)
        >>> it = list(range(1, 17, 1))
        >>> A = list(f(it)); B = list(f(iter(it)));  # trying the function on it (a list) and iter(it) (and iterator)
        >>> assert A == B  # it and iter(it) should give the same thing!
        >>> A  # and that thing is:
        [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], [7, 8, 9, 10, 11, 12, 13, 14, 15, 16], [10, 11, 12, 13, 14, 15, 16], [13, 14, 15, 16], [16]]

        Testing chk_step > chk_size with return_tail=TRUE, stop and start_at PRESENT
        and stop_at SMALLER than the largest index of it

        >>> f = lambda it: fixed_step_chunker(it, chk_size=3, chk_step=4, start_at=1, stop_at=7, return_tail=True)
        >>> it = list(range(1, 17, 1))
        >>> A = list(f(it)); B = list(f(iter(it)));  # trying the function on it (a list) and iter(it) (and iterator)
        >>> assert A == B  # it and iter(it) should give the same thing!
        >>> A  # and that thing is:
        [[2, 3, 4], [6, 7]]

        Testing chk_step > chk_size with return_tail=FALSE, stop and start_at PRESENT
        and stop_at SMALLER than the largest index of it

        >>> f = lambda it: fixed_step_chunker(it, chk_size=3, chk_step=4, start_at=1, stop_at=7, return_tail=False)
        >>> it = list(range(1, 17, 1))
        >>> A = list(f(it)); B = list(f(iter(it)));  # trying the function on it (a list) and iter(it) (and iterator)
        >>> assert A == B  # it and iter(it) should give the same thing!
        >>> A  # and that thing is:
        [[2, 3, 4]]

        Testing chk_step > chk_size with return_tail=FALSE, stop and start_at NOT PRESENT
        
        >>> f = lambda it: fixed_step_chunker(it, chk_size=3, chk_step=4, start_at=None, stop_at=None, return_tail=False)
        >>> it = list(range(1, 17, 1))
        >>> A = list(f(it)); B = list(f(iter(it)));  # trying the function on it (a list) and iter(it) (and iterator)
        >>> assert A == B  # it and iter(it) should give the same thing!
        >>> A  # and that thing is:
        [[1, 2, 3], [5, 6, 7], [9, 10, 11], [13, 14, 15]]

        Testing chk_step > chk_size with return_tail=TRUE, stop and start_at NOT PRESENT

        >>> f = lambda it: fixed_step_chunker(it, chk_size=3, chk_step=4, start_at=None, stop_at=None, return_tail=True)
        >>> it = list(range(1, 19, 1))
        >>> A = list(f(it)); B = list(f(iter(it)));  # trying the function on it (a list) and iter(it) (and iterator)
        >>> assert A == B  # it and iter(it) should give the same thing!
        >>> A  # and that thing is:
        [[1, 2, 3], [5, 6, 7], [9, 10, 11], [13, 14, 15], [17, 18]]

        Testing chk_step > chk_size with return_tail=TRUE, stop and start_at NOT PRESENT
        with negative values in the iterator

        >>> f = lambda it: fixed_step_chunker(it, chk_size=3, chk_step=4, start_at=None, stop_at=None, return_tail=True)
        >>> it = list(range(-10, 19, 1))
        >>> A = list(f(it)); B = list(f(iter(it)));  # trying the function on it (a list) and iter(it) (and iterator)
        >>> assert A == B  # it and iter(it) should give the same thing!
        >>> A  # and that thing is:
        [[-10, -9, -8], [-6, -5, -4], [-2, -1, 0], [2, 3, 4], [6, 7, 8], [10, 11, 12], [14, 15, 16], [18]]

        Testing chk_step > chk_size with return_tail=TRUE, stop and start_at NOT PRESENT
        with items of various types in the iterator

        >>> f = lambda it: fixed_step_chunker(it, chk_size=3, chk_step=2, start_at=None, stop_at=None, return_tail=True)
        >>> it = ['a', 3, -10, 9.2, str, [1,2,3], set([10,20])]
        >>> A = list(f(it)); B = list(f(iter(it)));  # trying the function on it (a list) and iter(it) (and iterator)
        >>> assert A == B  # it and iter(it) should give the same thing!
        >>> A  # and that thing is:
        [['a', 3, -10], [-10, 9.2, <class 'str'>], [<class 'str'>, [1, 2, 3], {10, 20}], [{10, 20}]]
       """

    if chk_step is None:
        chk_step = chk_size

    if start_at is None:
        start_at = 0

    # if the input is a list
    if hasattr(it, '__getitem__') and hasattr(it, '__len__'):
        if stop_at is None:
            stop_at = len(it)
        else:
            stop_at = min(len(it), stop_at)

        it = it[start_at:stop_at]
        it_minus_chk_length = len(it) - chk_size
        n_full_chk_to_return = max(int((it_minus_chk_length / chk_step) + 1), 0)
        bt = 0
        tt = bt + chk_size

        for i in range(n_full_chk_to_return):
            yield it[bt:tt]
            bt += chk_step
            tt += chk_step

        if return_tail:
            while len(it[bt:tt]) > 0:
                yield it[bt:tt]
                bt += chk_step
                tt += chk_step

    # if the input is an iterator
    else:
        if chk_step < chk_size:

            it = islice(it, start_at, stop_at)
            chk = list(islice(it, chk_size))

            while len(chk) == chk_size:
                yield chk
                chk = chk[chk_step:] + list(islice(it, chk_step))

        else:
            it = islice(it, start_at, stop_at)
            chk = list(islice(it, chk_size))
            gap = chk_step - chk_size

            while len(chk) == chk_size:
                yield chk
                chk = list(islice(it, gap, gap + chk_size))

        if return_tail:
            while len(chk) > 0:
                yield chk
                chk = chk[chk_step:]
