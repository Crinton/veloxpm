import os
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext as _build_ext


ROOT_DIR = Path(__file__).resolve().parent
PACKAGE_NAME = "veloxpm"
EXTENSION_BASENAME = "_veloxpm_core"


class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str) -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(_build_ext):
    def run(self) -> None:
        try:
            subprocess.check_call(["cmake", "--version"])
        except FileNotFoundError as exc:
            raise RuntimeError("CMake must be installed to build veloxpm.") from exc

        for ext in self.extensions:
            if isinstance(ext, CMakeExtension):
                self.build_cmake_extension(ext)
            else:
                super().build_extension(ext)

    def build_cmake_extension(self, ext: CMakeExtension) -> None:
        extdir = Path(self.get_ext_fullpath(ext.name)).parent.resolve()
        extdir.mkdir(parents=True, exist_ok=True)

        build_temp = Path(self.build_temp) / ext.name
        build_temp.mkdir(parents=True, exist_ok=True)

        debug = bool(self.debug or int(os.environ.get("DEBUG", "0")))
        cfg = "Debug" if debug else "Release"

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
        ]

        if sys.platform == "win32":
            cmake_args.extend(["-G", "Visual Studio 17 2022", "-A", "x64"])

        subprocess.check_call(["cmake", ext.sourcedir] + cmake_args, cwd=build_temp)
        subprocess.check_call(
            ["cmake", "--build", ".", "--config", cfg, "-j", str(os.cpu_count() or 4)],
            cwd=build_temp,
        )


setup(
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    include_package_data=True,
    ext_modules=[CMakeExtension(f"{PACKAGE_NAME}.{EXTENSION_BASENAME}", str(ROOT_DIR))],
    cmdclass={"build_ext": CMakeBuild},
)
