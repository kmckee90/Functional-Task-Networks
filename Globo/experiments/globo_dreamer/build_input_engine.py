"""Build a pinned Craftium input fix on a pod, outside source sync."""
import json
import os
from pathlib import Path
import subprocess
import argparse

# Wheel packaging revision, Luanti 5.12.0. The v0.0.1 git tag predates the
# rebuilt 0.0.1 wheel and points to an older engine, so do not build that tag.
COMMIT = "da75c9b7f9c429c5f0b96e25ae29ed1f74b908f4"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--single-place', action='store_true')
    args = parser.parse_args()
    state = Path(os.environ.get("GLOBO_STATE", "/home/dev/globo_dreamer_state"))
    source = state / "vendor/craftium-input-fix-512"
    if not source.exists():
        subprocess.run(["git", "clone", "--no-checkout", "--depth", "1",
                        "https://github.com/mikelma/craftium.git", str(source)], check=True)
    git = ["git", "-c", f"safe.directory={source.resolve()}", "-C", str(source)]
    revision = subprocess.check_output(git + ["rev-parse", "HEAD"], text=True).strip()
    if revision != COMMIT:
        subprocess.run(git + ["fetch", "--depth", "1", "origin", COMMIT], check=True)
        subprocess.run(git + ["checkout", "--detach", COMMIT], check=True)

    def patch(file, replacements):
        # Reconstruct from the pinned upstream version, making rebuilds idempotent.
        text = subprocess.check_output(git + ["show", f"{COMMIT}:{file}"], text=True)
        for old, new in replacements:
            if text.count(old) != 1:
                raise RuntimeError(f"Patch context mismatch: {file}: {old!r}")
            text = text.replace(old, new)
        (source / file).write_text(text)

    patch("src/client/craftium.h", [
        ("inline bool virtual_key_presses[KeyType::INTERNAL_ENUM_COUNT];",
         "inline bool virtual_key_presses[KeyType::INTERNAL_ENUM_COUNT];\n"
         "inline bool virtual_key_pressed[KeyType::INTERNAL_ENUM_COUNT] = {};\n"
         "inline bool virtual_key_released[KeyType::INTERNAL_ENUM_COUNT] = {};")])
    patch("src/client/inputhandler.h", [
        ("|| joystick.wasKeyPressed(k);", "|| joystick.wasKeyPressed(k) || virtual_key_pressed[k];"),
        ("|| joystick.wasKeyReleased(k);", "|| joystick.wasKeyReleased(k) || virtual_key_released[k];"),
        ("m_receiver->clearWasKeyPressed();", "m_receiver->clearWasKeyPressed();\n"
         "\t\tfor (auto &value : virtual_key_pressed) value = false;"),
        ("m_receiver->clearWasKeyReleased();", "m_receiver->clearWasKeyReleased();\n"
         "\t\tfor (auto &value : virtual_key_released) value = false;")])
    patch("src/client/client.cpp", [
        ("&& i != KeyType::DIG) {", "&& i != KeyType::DIG\n"
         + ("" if args.single_place else "\t\t\t\t&& i != KeyType::PLACE\n")
         + "\t\t\t\t&& i != KeyType::SNEAK) {"),
        ("/* Clear missing virtual key presses (WASD + L click) */",
         "const bool previous_dig = virtual_key_presses[KeyType::DIG];\n"
         "    const bool previous_place = virtual_key_presses[KeyType::PLACE];\n"
         "\t/* Clear missing virtual key presses (WASD + L click) */"),
        ("virtual_key_presses[KeyType::PLACE] = actions[9];",
         "virtual_key_presses[KeyType::PLACE] = actions[9];\n"
         "    // Each policy command supplies one click edge, held for its frameskip.\n"
         "    virtual_key_pressed[KeyType::DIG] = actions[8] != 0;\n"
         "    virtual_key_pressed[KeyType::PLACE] = actions[9] != 0;\n"
         "    virtual_key_released[KeyType::DIG] = previous_dig && !actions[8];\n"
         "    virtual_key_released[KeyType::PLACE] = previous_place && !actions[9];")])
    build = state / "build/craftium-input-v2-512"
    subprocess.run(["cmake", "-S", str(source), "-B", str(build), "-G", "Ninja",
                    "-DRUN_IN_PLACE=TRUE", "-DCMAKE_BUILD_TYPE=Release", "-DENABLE_SOUND=OFF",
                    "-DENABLE_GETTEXT=OFF", "-DENABLE_REDIS=OFF", "-DBUILD_UNITTESTS=OFF"], check=True)
    subprocess.run(["cmake", "--build", str(build), "-j", os.environ.get("BUILD_THREADS", "8")], check=True)
    import hashlib
    import shutil
    output = state / ('engines/input-v2-single-place' if args.single_place else 'engines/input-v2')
    output.mkdir(parents=True, exist_ok=True)
    binary = source / "bin/luanti"
    if not binary.exists():
        binary = build / "bin/luanti"
    shutil.copy2(binary, output / "luanti.new")
    (output / "luanti.new").replace(output / "luanti")
    (output / "manifest.json").write_text(json.dumps(dict(
        version=2, single_place=args.single_place, upstream=COMMIT,
        sha256=hashlib.sha256((output / "luanti").read_bytes()).hexdigest()), indent=2))
    print(f"Installed patched engine: {output / 'luanti'}", flush=True)


if __name__ == "__main__":
    main()
