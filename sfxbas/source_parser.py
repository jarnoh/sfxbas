"""Pair-aware source alternatives for optional final-build candidates.

Preserve several source-distance classes and bounded cost/reach alternatives.
The existing parser remains a separate candidate; this is not exhaustive DP.
"""
import bisect
from collections import deque
from functools import lru_cache
from . import matcher as m

_INF = float('inf')


class ActiveSources:
    """Active ranks in 256-bit blocks, indexed by a bitset of nonempty blocks."""
    def __init__(self,n):
        self.blocks=[0]*(n//256+1)
        self.nonempty=0

    def add(self,rank):
        block=rank>>8
        self.blocks[block]|=1<<(rank&255)
        self.nonempty|=1<<block

    def neighbours(self,rank):
        """Nearest active rank below rank, then at or above rank."""
        block,offset=rank>>8,rank&255
        bits=self.blocks[block]
        lower=bits&((1<<offset)-1)
        result=[]
        if lower:result.append((block<<8)+lower.bit_length()-1)
        else:
            blocks=self.nonempty&((1<<block)-1)
            if blocks:
                previous=blocks.bit_length()-1
                result.append((previous<<8)+self.blocks[previous].bit_length()-1)
        upper=bits>>offset
        if upper:result.append(rank+(upper&-upper).bit_length()-1)
        else:
            blocks=self.nonempty>>(block+1)
            if blocks:
                following=block+(blocks&-blocks).bit_length()
                bits=self.blocks[following]
                result.append((following<<8)+(bits&-bits).bit_length()-1)
        return result


@lru_cache(maxsize=32768)
def query_nodes(lo,hi):
    """Segment nodes in MinTree's traversal order for a 256-slot query."""
    lo+=256;hi+=256
    nodes=[]
    while lo<hi:
        if lo&1:nodes.append(lo);lo+=1
        if hi&1:hi-=1;nodes.append(hi)
        lo>>=1;hi>>=1
    return tuple(nodes)


class PairTrees:
    """Two 256-slot minima sharing a clear traversal and leaf payloads.

    As in MinTree, clear selects the left child on ties and add retains the
    existing winner. Queries never cover all 256 slots, so root 1 is unused.
    """
    def __init__(self):
        self.values=([_INF]*512,[_INF]*512)
        self.winners=([0]*512,[0]*512)
        self.states=[None]*256

    def set(self,slot,cost=_INF,state=None,offset=0):
        a,b=self.values
        wa,wb=self.winners
        i=slot+256
        a[i]=b[i]=_INF
        i>>=1
        while i>=2:
            left=2*i;right=left+1
            child=left if a[left]<=a[right] else right
            a[i]=a[child];wa[i]=wa[child]
            child=left if b[left]<=b[right] else right
            b[i]=b[child];wb[i]=wb[child]
            i>>=1
        self.states[slot]=state
        if state is None:return
        for values,winners,c in ((a,wa,cost),(b,wb,cost-offset)):
            i=slot+256
            values[i]=c;winners[i]=slot
            i>>=1
            while i>=2 and c<values[i]:
                values[i]=c;winners[i]=slot
                i>>=1

    def min(self,which,lo,hi):
        values=self.values[which]
        nodes=query_nodes(lo,hi)
        if not nodes:return _INF,None
        at=min(nodes,key=values.__getitem__)
        best=values[at]
        return best,self.states[self.winners[which][at]] if best<_INF else None


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
    # Distance classes repeatedly compare the same byte strings. This cache is
    # local to discovery and released with its keys when this call returns.
    prefix=lru_cache(maxsize=None)(m.common_prefix)
    result = []
    for distance in distances:
        active = ActiveSources(len(entries))
        lengths, sources = [0]*(n+1), [0]*(n+1)
        x = n
        for p in range(n,0,-1):
            while x >= p+distance:
                if x in ranks: active.add(ranks[x])
                x -= 1
            best = (0,0)
            for k in active.neighbours(qranks[p]):
                length = prefix(queries[p],keys[k])
                best = max(best,(length,entries[k][1]))
            lengths[p], sources[p] = best[0], best[1]-best[0]
        result.append((tuple(lengths),tuple(sources)))
    return tuple(result)


@lru_cache(maxsize=12)
def class_variants(t,ok):
    """Lambda-independent source alternatives, deduplicated in discovery order."""
    classes=source_classes(t,ok)
    return ((),)+tuple(tuple(dict.fromkeys((p-l[p],'match',s[p]+l[p])
                                          for l,s in classes if l[p]))
                       for p in range(1,len(t)+1))


@lru_cache(maxsize=32)
def paired(t,lam,base,ok,extra,rom,stop,pool=(),cycles=True):
    """Bounded multi-arrival DP; ring trees bound active-state memory.

    This remains a heuristic: three starts per source class, not all Pareto
    states. Every selected source is exact and every pair observes reach.
    """
    n = len(t)
    end = n if stop is None else stop
    choices=m.with_pool(m.matches(t,base,ok,extra,rom),pool)
    lengths,sources,kinds=(tuple(values[:end+1]) for values in choices)
    return paired_choices(t,lam,ok,end,lengths,sources,kinds,cycles)


@lru_cache(maxsize=32)
def paired_choices(t,lam,ok,end,lengths,sources,kinds,cycles):
    """Share a DP when staging plans yield identical effective copy choices."""
    n=len(t)
    # All statements have the same secondary cycle cost. Count statements for
    # ties instead of carrying 7099 cycles each: smaller exact integers preserve
    # the ordering. Without cycle ties, only the rational primary cost is needed.
    numerator,denominator = float(lam).as_integer_ratio()
    scale = 1 << (n+1).bit_length() if cycles else 1
    unit,numerator = denominator*scale,numerator*scale
    statement = 6*unit+2*numerator+int(cycles)
    variants_by_end = class_variants(t,ok)
    ring = 256
    # Queries cover at most 255 slots, so the 256-slot root is never read.
    trees = PairTrees()
    groups,expiry = {},[[] for _ in range(ring)]
    f,how = [0]+[_INF]*end,[None]*(end+1)
    lit,win = deque(),deque()
    def update(b,p):
        group=groups[b]
        while group and group[-1][1]<p:group.pop()
        if group:
            cost,limit,state=group[-1]
            trees.set(b%ring,cost,state,b*unit)
            expiry[(limit+1)%ring].append(b)
        else:trees.set(b%ring)
    def query(tree,lo,hi):
        lo,hi = lo%ring,hi%ring
        if lo<hi:return trees.min(tree,lo,hi)
        best = trees.min(tree,lo,ring)
        if hi:
            c,s = trees.min(tree,0,hi)
            if c<best[0]:best=c,s
        return best
    def starts(q,low):
        i = bisect.bisect_left(q,low)
        # Cheapest, next undominated, and latest: trade cost for pairing reach.
        last = len(q)-1
        first = q[i]
        if i == last: return (first,)
        second = q[i+1]
        if i+1 == last: return first,second
        return first,second,q[last]
    for p in range(1,end+1):
        # Every state expires within 255 endpoints. Sorting each bucket keeps
        # the heap's (expiry, endpoint) order, including its tree tie effects.
        due=expiry[p%ring]
        expiry[p%ring]=[]
        for b in sorted(due):
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
        c,s=query(1,max(0,p-255),p)
        if c+p*unit<best:
            a,b,k,src=s
            best,h=c+p*unit,('pair',a,b,k,src,'lit',0)
        if lengths[p]:
            c,s=query(0,p-lengths[p],p)
            if c<best:
                a,b,k,src=s
                best,h=c,('pair',a,b,k,src,kinds[p],sources[p]+lengths[p]-(p-b))
        f[p],how[p]=best,h
        if p==n:break
        groups.pop(p-ring,None)
        variants=[(p-min(p,255),'lit',0)]
        if lengths[p]:variants.append((p-lengths[p],kinds[p],sources[p]+lengths[p]))
        variants += variants_by_end[p]
        used=set();group=[]
        # Distance classes often choose the same source. Expand it only once;
        # dict order preserves the candidate order used to break equal costs.
        for low,k,top in dict.fromkeys(variants):
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
    parts=[];p=end
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
