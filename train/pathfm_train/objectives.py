"""The velocity flow-matching and on-policy Euler-label PDD objectives."""

import torch
from pathfm.sampling import draw_source

from .model import parallel_heads


def draw_inputs(model, tensors, recipe, stage, generators):
    draws = recipe["flow_draws"]
    rows, horizon = tensors["target"].shape
    device = tensors["target"].device
    sources = draw_source(
        rows,
        draws,
        horizon,
        sampling="iid",
        device=device,
        generator=generators["noise"],
        scale=recipe[stage]["training_source_scale"],
    ).permute(1, 0, 2)
    groups = tensors["groups"]
    if stage != "pdd":
        # Preserve the original stream's uniform draw before the logit-normal draw.
        torch.rand(draws * groups, generator=generators["time"], device=device)
        normal = torch.randn(
            draws * groups, generator=generators["time"], device=device
        )
        times = torch.sigmoid(-0.8 + 0.8 * normal).reshape(draws, groups)
        return {"sources": sources, "times": times[:, tensors["group_ids"]]}
    shape = (draws, groups)
    uniform = torch.rand((2, *shape), generator=generators["time"], device=device)
    lengths = torch.tensor(sorted(model.parallel_block_menu), device=device)
    choice = torch.rand(shape, generator=generators["time"], device=device)
    length = lengths[(choice * len(lengths)).long().clamp(max=len(lengths) - 1)]
    blocks = model.parallel_grid_size // length
    starts = (uniform[0] * blocks).long().clamp(max=blocks - 1) * length
    offsets = (uniform[1] * length).long().clamp(max=length - 1)
    return {
        "sources": sources,
        "starts": starts[:, tensors["group_ids"]],
        "intervals": (starts + offsets)[:, tensors["group_ids"]],
    }


def flow_loss(model, static, target, inputs, amp):
    normalized = model.normalize_target(target, static["loc_scale"])
    unknown = model.unknown_target_mask(static, torch.ones_like(target))
    sources, times = inputs["sources"], inputs["times"]
    states = model.blend_state(
        (1.0 - times.unsqueeze(-1)) * sources
        + times.unsqueeze(-1) * normalized.unsqueeze(0),
        static,
        unknown,
    )
    velocity_target = unknown.unsqueeze(0) * (normalized.unsqueeze(0) - sources)
    packed = model.prepare_packed_draws(static, sources.shape[0])
    with torch.autocast(
        device_type=target.device.type, dtype=torch.bfloat16, enabled=amp
    ):
        prediction = model.forward_packed(states, times, packed).float()
    total = (unknown.unsqueeze(0) * (prediction - velocity_target).square()).sum()
    return total / (sources.shape[0] * unknown.sum().clamp_min(1.0))


def pdd_loss(student, teacher, static, teacher_static, target, inputs, amp):
    sources, starts, intervals = (
        inputs["sources"],
        inputs["starts"],
        inputs["intervals"],
    )
    draws, rows = sources.shape[:2]
    knots = torch.tensor(
        student.parallel_knots, device=target.device, dtype=torch.float32
    )
    widths = knots[1:] - knots[:-1]
    normalized = student.normalize_target(target, static["loc_scale"])
    unknown = student.unknown_target_mask(static, torch.ones_like(target))
    times = knots[starts]
    states = student.blend_state(
        (1.0 - times.unsqueeze(-1)) * sources
        + times.unsqueeze(-1) * normalized.unsqueeze(0),
        static,
        unknown,
    )
    packed = student.prepare_packed_draws(static, draws)
    with torch.autocast(
        device_type=target.device.type, dtype=torch.bfloat16, enabled=amp
    ):
        heads = parallel_heads(student, states, times, packed).float()
    with torch.no_grad():
        grid = torch.arange(student.parallel_grid_size, device=target.device)
        inside = (grid >= starts.unsqueeze(-1)) & (grid < intervals.unsqueeze(-1))
        displacement = torch.einsum(
            "drn,drnf->drf", widths * inside.to(widths.dtype), heads.detach()
        )
        label_states = student.blend_state(states + displacement, static, unknown)
        teacher_packed = teacher.prepare_packed_draws(teacher_static, draws)
        with torch.autocast(
            device_type=target.device.type, dtype=torch.bfloat16, enabled=amp
        ):
            labels = teacher.forward_packed(
                label_states, knots[intervals], teacher_packed
            ).float()
        labels = unknown.unsqueeze(0) * labels
    index = intervals.reshape(draws, rows, 1, 1).expand(draws, rows, 1, heads.shape[-1])
    prediction = heads.gather(2, index).squeeze(2)
    squared = (unknown.unsqueeze(0) * (prediction - labels)).square().sum(dim=-1)
    return squared.sum() / (draws * unknown.sum().clamp_min(1.0))


class Objective(torch.nn.Module):
    def __init__(self, model, teacher=None, amp=True):
        super().__init__()
        self.model = model
        object.__setattr__(self, "teacher", teacher)
        self.amp = amp

    def forward(self, tensors, inputs):
        context, target = tensors["context"], tensors["target"]
        with torch.autocast(
            device_type=context.device.type, dtype=torch.bfloat16, enabled=self.amp
        ):
            static = self.model.prepare_static(context, group_ids=tensors["group_ids"])
        if self.teacher is None:
            return flow_loss(self.model, static, target, inputs, self.amp)
        with (
            torch.no_grad(),
            torch.autocast(
                device_type=context.device.type, dtype=torch.bfloat16, enabled=self.amp
            ),
        ):
            teacher_static = self.teacher.prepare_static(
                context, group_ids=tensors["group_ids"]
            )
        return pdd_loss(
            self.model, self.teacher, static, teacher_static, target, inputs, self.amp
        )
