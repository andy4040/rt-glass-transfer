"""CPU-only guards for the audited, pinned RT training adaptation.

The digest is for RT-Splatting train.py at commit
3f45b3cac4be04db9f3092234666b695991b268a, with universal newlines. CUDA header
compatibility patches do not change that Python source.
"""
import hashlib


UPSTREAM_TRAIN_SHA256 = '01013e4c33bb18bcae2eeb8cc895515fcbf7a8a6bea18ebb7278fb5028f0cacf'


def reject_checkpoint_resume(argv):
    """Reject upstream's incomplete checkpoint resume before CUDA is imported."""
    for argument in argv:
        option = argument.split('=', 1)[0]
        # ArgumentParser also accepts unambiguous long-option abbreviations.
        if option.startswith('--') and len(option) > 2 and '--start_checkpoint'.startswith(option):
            raise ValueError(
                'RT checkpoint resume is unsupported: upstream capture/restore omits '
                'material parameters and shading networks. Use a new output directory; '
                'saved PLY/PT models remain valid for evaluation.'
            )


def replace_exact(source, old, new):
    count = source.count(old)
    if count != 1:
        raise ValueError(f'Expected exactly one RT source anchor, found {count}: {old!r}')
    return source.replace(old, new, 1)


def patch_training_source(source):
    """Fail closed if the upstream training code is no longer the audited source."""
    source = source.replace('\r\n', '\n').replace('\r', '\n')
    digest = hashlib.sha256(source.encode('utf-8')).hexdigest()
    if digest != UPSTREAM_TRAIN_SHA256:
        raise ValueError(f'Unaudited RT train.py source: SHA256 {digest}; expected {UPSTREAM_TRAIN_SHA256}')
    start_anchor = '        if opt.mask_loss_weight > 0 and iteration >= opt.mask_loss_from_iter:'
    end_anchor = '        total_loss = loss'
    # Check cardinality even with the source digest pinned, so adaptations remain
    # explicit if the supported upstream version is deliberately updated.
    for anchor in (start_anchor, end_anchor):
        if source.count(anchor) != 1:
            raise ValueError(f'Expected exactly one RT source anchor: {anchor!r}')
    start, end = source.index(start_anchor), source.index(end_anchor)
    if end <= start:
        raise ValueError('RT mask-loss anchors are out of order')
    source = source[:start] + '''        terms = mask_terms(surface_opacity, transmissivity, render_scat, gt_transparent_mask, viewpoint_cam.mask_valid.cuda(), opt, iteration)
        for name, term in terms.items():
            loss += term
            loss_dict[name] = float(term.detach())

''' + source[end:]
    source = replace_exact(source,
        'data_idx = np.random.randint(len(viewpoint_stack))',
        'data_idx = int(view_rng.integers(len(viewpoint_stack)))')
    source = replace_exact(source,
        '    gaussians.training_setup(opt)',
        '    gaussians.training_setup(opt)\n    initial_hash(gaussians, dataset.model_path)')
    source = replace_exact(source,
        '            iter_end.synchronize()',
        '            iter_end.synchronize()\n            record(gaussians, iteration, total_loss, dataset.model_path)')
    # Use only the independent evaluator; never upstream's padded window PSNR.
    source = replace_exact(source,
        '            training_report(tb_writer, tb_executor, opt, iteration, loss, loss_dict, iter_start.elapsed_time(iter_end), testing_iterations, scene, partial(render, pipe=pipe, bg_color=bg))',
        '            pass  # independent cam00 evaluator runs outside training')
    return source
