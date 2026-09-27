# Geometry reliability screen

Desktop-owned question: can the accepted 2D and distance-angle EdgeState
predictions be combined to reduce development error without new training?

`protocol.md` freezes the local screen. `screen.py` reads only the two accepted
prediction payloads and emits a row-aligned result. The terminal decision and
RML records remain separate from the historical training pair.
