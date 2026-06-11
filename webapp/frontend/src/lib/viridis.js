// 11-stop viridis approximation; t in [0,1] -> [r,g,b] 0..1
const STOPS = [
  [0.267,0.005,0.329],[0.283,0.141,0.458],[0.254,0.265,0.530],
  [0.207,0.372,0.553],[0.164,0.471,0.558],[0.128,0.567,0.551],
  [0.135,0.659,0.518],[0.267,0.749,0.441],[0.478,0.821,0.318],
  [0.741,0.873,0.150],[0.993,0.906,0.144],
]
export function viridis(t) {
  t = Math.max(0, Math.min(1, t)) * (STOPS.length - 1)
  const i = Math.floor(t), f = t - i
  const a = STOPS[i], b = STOPS[Math.min(i + 1, STOPS.length - 1)]
  return [a[0]+(b[0]-a[0])*f, a[1]+(b[1]-a[1])*f, a[2]+(b[2]-a[2])*f]
}
