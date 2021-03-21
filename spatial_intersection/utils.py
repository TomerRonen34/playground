import re
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Union

import matplotlib.pyplot as plt
from reportlab.graphics.renderPM import drawToPIL
from shapely.geometry import GeometryCollection
from shapely.geometry.base import BaseGeometry
from svglib.svglib import svg2rlg


def vis_geometries_as_html(geometry_collection: GeometryCollection, path: Union[Path, str]) -> None:
    svg_string = geometry_collection._repr_svg_()
    svg_string = re.sub(r'width="\d+\.?\d*"', 'width="700"', svg_string, count=1)
    svg_string = re.sub(r'height="\d+\.?\d*"', 'height="700"', svg_string, count=1)
    with open(path, 'w') as f:
        f.write(svg_string)


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
