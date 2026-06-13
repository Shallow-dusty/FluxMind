"""Editable no-key local execution templates for control-engineering jobs."""

PYTHON_EXECUTION_TEMPLATES = {
    "hello": "print('fluxmind job ok')",
    "smc_reaching_law": """from pathlib import Path
import math

eta = 1.8
lam = 3.0
dt = 0.01
steps = 240
s = 1.2
rows = []
for k in range(steps + 1):
    t = k * dt
    rows.append((t, s))
    sign = 1 if s > 0 else -1 if s < 0 else 0
    s += dt * (-eta * sign - lam * s)

csv_lines = ['time_s,sliding_surface'] + [f'{t:.4f},{value:.6f}' for t, value in rows]
Path('smc_reaching_law.csv').write_text('\\n'.join(csv_lines) + '\\n', encoding='utf-8')

width, height = 720, 360
margin = 42
values = [value for _t, value in rows]
v_min, v_max = min(values), max(values)
span = max(v_max - v_min, 1e-9)
points = []
for index, (_t, value) in enumerate(rows):
    x = margin + index * (width - 2 * margin) / (len(rows) - 1)
    y = height - margin - (value - v_min) * (height - 2 * margin) / span
    points.append(f'{x:.1f},{y:.1f}')
svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="{width}" height="{height}" fill="#f8fafc"/>
  <line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#111827" stroke-width="2"/>
  <line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#111827" stroke-width="2"/>
  <polyline points="{' '.join(points)}" fill="none" stroke="#2563eb" stroke-width="4"/>
  <text x="{width/2}" y="28" text-anchor="middle" font-family="Arial" font-size="20" fill="#111827">SMC reaching-law response</text>
</svg>
'''
Path('smc_reaching_law.svg').write_text(svg, encoding='utf-8')
print('wrote smc_reaching_law.csv and smc_reaching_law.svg')
""",
    "pmsm_current_step": """from pathlib import Path
import math

i_ref = 1.0
wn = 22.0
zeta = 0.18
dt = 0.005
steps = 240
wd = wn * math.sqrt(max(1.0 - zeta * zeta, 0.0))
rows = []
for k in range(steps + 1):
    t = k * dt
    envelope = math.exp(-zeta * wn * t)
    iq = i_ref * (1.0 - envelope * math.cos(wd * t))
    rows.append((t, i_ref, iq, i_ref - iq))

csv_lines = ['time_s,iq_ref,iq,error'] + [
    f'{t:.4f},{ref:.6f},{iq:.6f},{err:.6f}' for t, ref, iq, err in rows
]
Path('pmsm_current_step.csv').write_text('\\n'.join(csv_lines) + '\\n', encoding='utf-8')

width, height = 720, 360
margin = 42
values = [iq for _t, _ref, iq, _err in rows]
v_min, v_max = min(values), max(values)
span = max(v_max - v_min, 1e-9)
points = []
for index, (_t, _ref, value, _err) in enumerate(rows):
    x = margin + index * (width - 2 * margin) / (len(rows) - 1)
    y = height - margin - (value - v_min) * (height - 2 * margin) / span
    points.append(f'{x:.1f},{y:.1f}')
svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="{width}" height="{height}" fill="#f8fafc"/>
  <line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#111827" stroke-width="2"/>
  <line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#111827" stroke-width="2"/>
  <polyline points="{' '.join(points)}" fill="none" stroke="#dc2626" stroke-width="4"/>
  <text x="{width/2}" y="28" text-anchor="middle" font-family="Arial" font-size="20" fill="#111827">PMSM q-axis current step response</text>
</svg>
'''
Path('pmsm_current_step.svg').write_text(svg, encoding='utf-8')
print('wrote pmsm_current_step.csv and pmsm_current_step.svg')
""",
}

OCTAVE_EXECUTION_TEMPLATES = {
    "hello": "disp('fluxmind octave job ok');",
    "pmsm_current_decay": """t = linspace(0, 1.2, 121)';
iq_ref = ones(size(t));
iq = 1 - exp(-5 .* t) .* cos(18 .* t);
error = iq_ref - iq;
csvwrite('pmsm_current_decay.csv', [t, iq_ref, iq, error]);
disp('wrote pmsm_current_decay.csv');
""",
    "smc_sign_switching": """t = linspace(0, 1.0, 101)';
s = 1.5 .* exp(-4 .* t) .* sign(cos(6 .* t));
u = -1.2 .* sign(s);
csvwrite('smc_sign_switching.csv', [t, s, u]);
disp('wrote smc_sign_switching.csv');
""",
}
