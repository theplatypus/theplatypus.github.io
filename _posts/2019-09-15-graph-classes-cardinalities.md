---
title: Some calculus about graph classes cardinalities
date: 2019-09-15 08:00:00 +0100
categories: [Data Structures and Algorithms, Graph Theory]
tags: [graph, isomorphism]     # TAG names should always be lowercase
---

```python
from math import factorial, ceil

def fact(n):
    """
    n!
    <just as a reminder, we do prefer using C implementation of math library therefore>
    """
    if n == 0 :
        return 1
    else :
        return n * fact(n-1)

def binomial(n, k):
    """
    ( n )          n!
    |   |   =   ————————
    ( k )       k!(n-k)!
    """
    if (k > n):
        return 0
    elif (k == 2):
        # little speed-up
        return (n * (n-1)) // 2
    else:
        return factorial(n) // (factorial(k) * factorial(n - k))

def G_n(n):
    """
    number of not necessarly connected, labelled graph 
    on n vertices

    Each possible edge either exists, either doesn't :
        2 states possible ^ number of possible edges

    note : binomial(n, 2) = (n*(n-1)//2)
    
    OEIS A006125 (1, 2, 8, 64, 1024, 32768, ... )
    >>> print(list([G_n(n) for n in range(0, 10)]))
    """
    return 2 ** binomial(n, 2)

def G_nk(n, k):
    """
    number of not necessarly connected, labelled graph 
    on n vertices, k edges

    Very naive, take k pairs among all possible pairs of edges
    """
    return binomial(binomial(n, 2), k)

def C_n(n):
    """
    number of connected, labelled graph 
    on n vertices

    note : logarithmic transform of OEIS A006125 (G_n)

    sources :
    - Harary and Palmer, p. 7
    - http://mathworld.wolfram.com/LabeledGraph.html

    OEIS A001187 (1, 1, 1, 4, 38, 728, 26704, ... )
    >>> print(list([C_n(n) for n in range(0, 10)]))
    """
    if n == 0:
        return 1
    else :
        return 2 ** binomial(n, 2) - sum([
            k * binomial(n, k) * 2** binomial(n-k, 2) * C_n(k) 
            for k in range(1, n)
        ]) // n

def T_n(n):
    """
    Number of free labelled trees on n vertices
    note : k = n - 1

    source : 
     - Cayley's formula https://en.wikipedia.org/wiki/Cayley%27s_formula

    OEIS A000272 (1, 1, 1, 3, 16, 125, 1296, 16807, 262144, ... )
    >>> print(list([T_n(n) for n in range(0, 10)]))
    """
    if (n < 1):
        return 1
    else:
        return int(n ** (n - 2))

def C_nk(n, k):
    """
    Number of connected, labelled graphs 
    on n vertices, k edges
    
    sources :
    - Enumeration of Labelled Graphs - E. N. Gilbert (Theorem II) : http://oeis.org/A001187/a001187.pdf
    - https://math.stackexchange.com/questions/689526/how-many-connected-graphs-over-v-vertices-and-e-edges

    >>> [ (C_nk(n, k), k) for k in range( n - 1, binomial(n, 2)) ]
    """
    if k < n - 1 or k > binomial(n, 2) :
        return 0
    elif k == n - 1:
        # T_n(n) = n ** (n-2)
        return T_n(n)
    else:
        # G_nk(n, k) = binomial(binomial(n, 2), k)
        return G_nk(n, k) - sum([
            binomial(n - 1, m) * sum([
                binomial(((n - 1 - m) * (n - 2 - m) // 2), p) * C_nk(m + 1, k - p)
                for p in range(0, k)
            ])
            for m in range(0, n - 2)
         ])

def Ck_n(n):
    """
    number of connected, labelled graphs
    on n vertices, for each k fixed

    >>> [ max(Ck_n(n))[1] for n in range(3, 12) ] gives OEIS A054925
    """
    return [ (C_nk(n, k), k) for k in range( n - 1, binomial(n, 2)) ]

def MAX_Ck_n(n):
    """
    maximum number of labelled, connected graph
    on n vertices, but with a fixed k

    k_max is half the number of possible edges, as central polynomial coef. states
        binomial(n, p) is max when n = 2p
        ceil helps to find an int where we are in a case where max coef occurs twice

    => k_max = (n, 2) / (2 = n(n-1) / 2) /2 = n(n-1) / 4

    >>> [ MAX_Ck_n(n) for n in range(0, 10)] 
    """
    return C_nk(n, ceil((n*(n-1)) / 4))
```