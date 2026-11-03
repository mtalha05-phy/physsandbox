"""Containment demo: runaway memory use. Must be killed by the memory limit."""
blocks = []
while True:
    blocks.append(bytearray(50_000_000))
