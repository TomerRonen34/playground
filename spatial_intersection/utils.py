from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
from reportlab.graphics.renderPM import drawToPIL, drawToPILP
from shapely.geometry.base import BaseGeometry
from svglib.svglib import svg2rlg


def plot_geometry(geom: BaseGeometry) -> None:
    svg_string = geom._repr_svg_()
    with TemporaryDirectory() as tmp_dir:
        tmp_path = str(Path(tmp_dir) / "geom.svg")
        with open(tmp_path, 'w') as f:
            f.write(svg_string)
        drawing_obj = svg2rlg(tmp_path)
    pil_image = drawToPIL(drawing_obj, configPIL={"alhpa": 0.5})
    plt.imshow(pil_image)
    plt.show()
