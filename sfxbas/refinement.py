"""Refine a selected parse using literal ownership and complete layout costs.

The decoder and stream format stay unchanged. Only simulator-verified complete
builds that improve the requested objective replace the original result.
"""
from dataclasses import replace
from . import matcher as m, packer, stream


def feedback_place(original):
    def place(job,parts,holes,paired,raw=None):
        best=original(job,parts,holes,paired,raw)
        current=list(parts)
        # Two feedback rounds. Proposals are only merges; no new decoder paths.
        for _ in range(2):
            needs=[(('l',p.start),p.length,0) for c in best.layout.chunks
                   for u,d in c.pairs for p,need in stream.literal_needs(u,d)]
            shared=stream.share_literals(job.t,needs)
            proposals=[]
            for i,p in enumerate(current):
                if p.kind=='skip':continue
                for count in (2,3,4):
                    group=current[i:i+count]
                    if len(group)!=count or any(x.kind=='skip' for x in group):break
                    end=group[-1].start+group[-1].length
                    length=end-p.start
                    if length>255:break
                    if any(a.start+a.length!=b.start for a,b in zip(group,group[1:])):break
                    options=[m.Part(p.start,length,'lit',0)]
                    # Join logical sources; simulation checks physical staging and source lifetime.
                    if all(x.kind==p.kind and x.src==p.src+x.start-p.start for x in group) and p.kind in ('match','rom'):
                        options.append(m.Part(p.start,length,p.kind,p.src))
                    stored=sum(x.length for x in group if x.kind=='lit' and x.start not in shared)
                    for merged in options:
                        extra=length if merged.kind=='lit' else 0
                        saving=3*(count-1)+stored-extra
                        # Include near misses: fewer statements/chunks can repay bytes.
                        if saving>=-4:
                            proposals.append((saving,count,i,merged))
            proposals.sort(key=lambda x:(x[0],x[1]),reverse=True)
            if not proposals:break
            trials=[]
            # A combined, non-overlapping proposal and the strongest individual edits.
            picked=[];occupied=set()
            for saving,count,i,merged in proposals:
                if saving<0:continue
                if not occupied.intersection(range(i,i+count)):
                    picked.append((i,count,merged));occupied.update(range(i,i+count))
            if picked:
                candidate=list(current)
                for i,count,merged in sorted(picked,reverse=True):candidate[i:i+count]=[merged]
                trials.append(candidate)
            for saving,count,i,merged in proposals[:6]:
                trials.append(current[:i]+[merged]+current[i+count:])
            improved=None
            for candidate in trials:
                try:r=original(job,candidate,holes,paired,raw)
                except packer.NoFit:continue
                if packer.rank(r,job.t,job.opts)<packer.rank(best,job.t,job.opts):
                    best,improved=r,candidate
            if improved is None:break
            current=improved
        return best
    return place


def refine_layout(t, result, opts):
    """Two local merge rounds on the selected staging plan, retaining result.

    Run after a lambda/size/speed search, so the bounded layout work is paid once
    per command rather than for every search candidate. This is local refinement,
    not a joint optimization of parsing, staging and layout.
    """
    opts = replace(opts, paired=result.layout.paired)
    job = packer.Job(packer.mask_color_ram(t), result.layout.stage,
                     result.layout.basic_end, opts, list(result.layout.fill_runs))
    parts = [p for p in result.parts if p.start >= 0]
    try:
        candidate = feedback_place(packer.place)(job, parts, result.layout.holes,
                                                 opts.paired, result.layout.raw)
    except packer.NoFit:
        return result
    if packer.rank(candidate, t, opts) >= packer.rank(result, t, opts):
        return result
    candidate.lam = result.lam
    candidate.cycle_ties = result.cycle_ties
    return candidate


def refine_result(t, result, opts, run=None):
    """Compare the original, layout feedback and a paired-source alternative.

    Source discovery runs once at the selected lambda, not at every search point.
    Raw builds and single parses retain the cheaper layout-only refinement.
    run: an ordered map for the alternative's independent build steps, such as the CLI's worker pool.
    """
    opts = replace(opts, paired=result.layout.paired)
    candidates = [result, refine_layout(t, result, opts)]
    if opts.paired and not opts.raw and result.layout.raw is None:
        try:
            source_opts = replace(opts, source_choices=True)
            alternative = packer.build(t, result.lam, source_opts, run=run)
            candidates.extend((alternative, refine_layout(t, alternative, source_opts)))
        except packer.NoFit:
            pass
    return min(candidates, key=lambda r: packer.rank(r, t, opts))
