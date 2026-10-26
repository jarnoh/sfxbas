"""Pair-aware source alternatives for optional final-build candidates.

Preserve several source-distance classes and bounded cost/reach alternatives.
The existing parser remains a separate candidate; this is not exhaustive DP.
"""
import bisect
from collections import deque
from functools import lru_cache
import heapq
from . import matcher as m, cycle_parser


@lru_cache(maxsize=12)
def source_classes(t, ok, edges=(), distances=(1, 32, 128, 255)):
    """Exact longest readable suffix in each minimum-distance class.

    Source keys stop at unreadable bytes and segment edges. Activate source
    tops from high to low; nearest active suffix neighbours suffice for longest
    prefix, including 1/2-byte matches. Unlike far_matches, no scan limit.
    """
    n = len(t)
    rr = [0] * (n+1)
    edges = set(edges)
    for x in range(1, n+1):
        rr[x] = min(255, (0 if x-1 in edges else rr[x-1])+1) if ok[x-1] else 0
    entries = sorted((t[x-rr[x]:x][::-1], x) for x in range(1, n+1) if rr[x])
    keys = [k for k, x in entries]
    ranks = {x: i for i, (k, x) in enumerate(entries)}
    queries = [b''] + [t[max(0,p-255):p][::-1] for p in range(1,n+1)]
    qranks = [bisect.bisect_left(keys,q) for q in queries]
    result = []
    for distance in distances:
        active = m.ActiveRanks(len(entries))
        lengths, sources = [0]*(n+1), [0]*(n+1)
        x = n
        for p in range(n,0,-1):
            while x >= p+distance:
                if x in ranks: active.add(ranks[x])
                x -= 1
            count = active.count(qranks[p])
            neighbours = ([active.kth(count)] if count else [])
            if count < active.total: neighbours.append(active.kth(count+1))
            best = (0,0)
            for k in neighbours:
                length = m.common_prefix(queries[p],keys[k])
                best = max(best,(length,entries[k][1]))
            lengths[p], sources[p] = best[0], best[1]-best[0]
        result.append((tuple(lengths),tuple(sources)))
    return tuple(result)


@lru_cache(maxsize=32)
def paired(t,lam,base,ok,extra,rom,stop,pool=(),cycles=True):
    """Bounded multi-arrival DP; ring trees bound active-state memory.

    This remains a heuristic: three starts per source class, not all Pareto
    states. Every selected source is exact and every pair observes reach.
    """
    n = len(t)
    unit,numerator,scale = cycle_parser.weights(lam,n)
    statement = 6*unit+2*numerator+(7099 if cycles else 0)
    lengths,sources,kinds = m.with_pool(m.matches(t,base,ok,extra,rom),pool)
    classes = source_classes(t,ok)
    ring = 256
    trees = m.MinTree(ring),m.MinTree(ring)
    groups,expiry = {},[]
    f,how = [0]+[float('inf')]*n,[None]*(n+1)
    lit,win = deque(),deque()
    def update(b,p):
        group=groups[b]
        while group and group[-1][1]<p:group.pop()
        for tree in trees:tree.clear(b%ring)
        if group:
            cost,limit,state=group[-1]
            trees[0].add(b%ring,(cost,state))
            trees[1].add(b%ring,(cost-b*unit,state))
            heapq.heappush(expiry,(limit+1,b))
    def query(tree,lo,hi,p):
        lo,hi = lo%ring,hi%ring
        ranges = [(lo,hi)] if lo<hi else [(lo,ring),(0,hi)]
        best = (float('inf'),None)
        for a,b in ranges:
            if a<b:
                c,s = tree.min(a,b)
                if c<best[0]:best=c,s
        return best
    def starts(q,low):
        i = bisect.bisect_left(q,low)
        # Cheapest, next undominated, and latest: trade cost for pairing reach.
        return list(dict.fromkeys(q[j] for j in (i,min(i+1,len(q)-1),len(q)-1)))
    for p in range(1,n+1):
        while expiry and expiry[0][0]<=p:
            _,b=heapq.heappop(expiry)
            if b in groups:update(b,p)
        j = p-1
        while lit and f[lit[-1]]-lit[-1]*unit >= f[j]-j*unit: lit.pop()
        while win and f[win[-1]] >= f[j]: win.pop()
        lit.append(j);win.append(j)
        while lit[0]<p-255:lit.popleft()
        while win[0]<p-255:win.popleft()
        la=lit[0]
        best,h=f[la]+(p-la)*unit+statement,('lone',la,'lit',0)
        if lengths[p]:
            ca=win[bisect.bisect_left(win,p-lengths[p])]
            c=f[ca]+statement
            if c<best:best,h=c,('lone',ca,kinds[p],sources[p]+lengths[p]-(p-ca))
        c,s=query(trees[1],max(0,p-255),p,p)
        if c+p*unit<best:
            a,b,k,src=s
            best,h=c+p*unit,('pair',a,b,k,src,'lit',0)
        if lengths[p]:
            c,s=query(trees[0],p-lengths[p],p,p)
            if c<best:
                a,b,k,src=s
                best,h=c,('pair',a,b,k,src,kinds[p],sources[p]+lengths[p]-(p-b))
        f[p],how[p]=best,h
        if p==n:break
        groups.pop(p-ring,None)
        variants=[(p-min(p,255),'lit',0)]
        if lengths[p]:variants.append((p-lengths[p],kinds[p],sources[p]+lengths[p]))
        variants += [(p-l[p],'match',s[p]+l[p]) for l,s in classes if l[p]]
        used=set();group=[]
        for low,k,top in variants:
            for a in starts(lit if k=='lit' and top==0 else win,low):
                # A pooled literal is a free copy, not a newly stored literal.
                literal = k=='lit' and top==0 and low==p-min(p,255)
                src=0 if k=='lit' else top-(p-a)
                limit=a+min(255,top-p) if k=='match' and top>p else a+255
                cost=f[a]+statement+((p-a)*unit if literal else 0)
                key=(a,k,src,cost,limit)
                if key in used:continue
                used.add(key)
                group.append((cost,limit,(a,p,k,src)))
        groups[p]=sorted(group,key=lambda x:(x[0],-x[1]),reverse=True)
        update(p,p+1)
    parts=[];p=n if stop is None else stop
    while p:
        h=how[p]
        if h[0]=='lone':
            _,a,k,src=h;parts.append(m.Part(a,p-a,k,src));p=a
        else:
            _,a,b,k,src,uk,usrc=h
            parts.extend((m.Part(b,p-b,uk,usrc),m.Part(a,b-a,k,src)));p=a
    return tuple(reversed(parts))


def legacy_paired(*args):
    return paired(*args, cycles=False)
