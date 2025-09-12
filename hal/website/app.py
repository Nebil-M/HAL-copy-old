import glob
import os
import re

import dash
import pandas as pd
import plotly.express as px
from dash import Input, Output, dcc, html

# 1. Load the data from your CSV
APP_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(APP_DIR, "data.csv")

try:
    df = pd.read_csv(CSV_PATH)
    print(f"Successfully loaded {len(df)} rows from {CSV_PATH}")
except FileNotFoundError:
    print(f"FATAL ERROR: Could not find data.csv at the expected path: {CSV_PATH}")
    df = pd.DataFrame(
        {
            "hover_label": [],
            "code_type": [],
            "nkd": [],
            "logical_efficiency": [],
            "weight": [],
            "num_layers": [],
            "avg_coupler_length": [],
            "max_avg_face_switches": [],
            "avg_tsvs_per_edge": [],
            "hardware_cost": [],
            "layout_dir_path": [],
        }
    )

# 2. Initialize the Dash app
app = dash.Dash(__name__, title="QEC Code Layouts")
server = app.server

# 3. Create the Plotly figure
code_type_colors = {
    "Surface code": "#000000",
    "Directional code": "#A7A7A7",
    "BB code": "#B97FC4",
    "Narrow BB code": "#B97FC4",
    "Radial code": "#FFBA0C",
    "Gross code": "#5992C5",
    "Tile code": "#EF6178",
    "Tanner code": "#A5DC6A",
}

fig = px.scatter(
    df,
    x="logical_efficiency",
    y="hardware_cost",
    color="code_type",
    color_discrete_map=code_type_colors,
    custom_data=["layout_dir_path", "weight", "code_type"],
    hover_name="hover_label",
    hover_data={
        "code_type": False,
        "nkd": False,
        "logical_efficiency": ":.2f",
        "weight": ":d",  # format as integer
        "num_layers": ":d",  # format as integer
        "avg_coupler_length": ":.2f",
        "max_avg_face_switches": ":.2f",
        "avg_tsvs_per_edge": ":.2f",
        "hardware_cost": ":.2f",
    },
    labels={
        "logical_efficiency": "Logical efficiency kd^2/n",
        "num_layers": "Tiers",
        "avg_coupler_length": "Length",
        "max_avg_face_switches": "Bump bonds",
        "avg_tsvs_per_edge": "TSVs",
        "hardware_cost": "Hardware cost C_hw",
    },
)

# Set alpha=0.5 for narrow codes
for trace in fig.data:
    if trace.name == "Narrow BB code":
        trace.opacity = 0.5

fig.update_layout(
    legend_title_text="",
    plot_bgcolor="white",
    xaxis={"gridcolor": "lightgrey", "range": [0, 30]},
    yaxis={"gridcolor": "lightgrey", "range": [0.8, 4.2]},
    legend={"x": 0.01, "y": 0.99, "bordercolor": "lightgrey", "borderwidth": 1},
)
fig.update_traces(marker={"size": 12, "line": {"width": 1, "color": "white"}})

# 4. Define the app layout
app.layout = html.Div(
    [
        html.H1(
            "Hardware-Aware Layouts (HAL) of Quantum Error Correcting Codes",
            style={"textAlign": "center"},
        ),
        html.P(
            "Click on a data point to view the corresponding layout.",
            style={"textAlign": "center"},
        ),
        html.Div(
            [
                dcc.Graph(
                    id="code-scatter-plot",
                    figure=fig,
                    style={"height": "80vh", "width": "50%"},
                    config={
                        "displayModeBar": True,
                        "modeBarButtonsToRemove": [
                            "pan2d",
                            "select2d",
                            "lasso2d",
                            "autoScale2d",
                        ],
                        "scrollZoom": True,  # allow scroll wheel to zoom
                        "displaylogo": False,
                    },
                ),
                # This Div will hold the gallery of layer images
                html.Div(
                    id="image-gallery-container",
                    style={
                        "width": "50%",
                        "padding": "10px",
                        "overflowY": "auto",
                        "height": "80vh",
                    },
                    children=[
                        html.P(
                            "Click a point on the graph to see its layout here.",
                            style={"textAlign": "center", "marginTop": "40vh"},
                        )
                    ],
                ),
            ],
            style={"display": "flex", "flexDirection": "row"},
        ),
    ]
)


@app.callback(
    Output("image-gallery-container", "children"),
    Input("code-scatter-plot", "clickData"),
)
def update_image_gallery(clickData):
    if clickData is None:
        return [
            html.P(
                "Click a point on the graph to see its layout here.",
                style={"textAlign": "center", "marginTop": "40vh"},
            )
        ]

    try:
        customdata = clickData["points"][0]["customdata"]
        browser_url_path = customdata[0]
        weight = customdata[1]
        code_type = customdata[2]
        hover_label = clickData["points"][0]["hovertext"]
    except KeyError:
        return html.P("Error: 'customdata' key not found in clickData.")

    # If surface code, display appropriate lattice text instead of image gallery
    if code_type in ["Surface code"]:
        lattice_type = "Square lattice" if weight == 4 else "Hex lattice"
        return html.Div(
            [
                html.H3(f"{code_type}", style={"textAlign": "center"}),
                html.P(
                    lattice_type,
                    style={
                        "textAlign": "center",
                        "fontSize": "24px",
                        "marginTop": "30vh",
                    },
                ),
            ]
        )

    server_fs_path = os.path.join(APP_DIR, browser_url_path)

    if not os.path.isdir(server_fs_path):
        return html.P(f"Error: Server could not find directory at: '{server_fs_path}'")

    escaped_path = glob.escape(server_fs_path)
    search_pattern = os.path.join(escaped_path, "*.png")

    def extract_layer_number(filename):
        match = re.search(r"(\d+)", filename)
        return int(match.group(1)) if match else float("inf")

    image_files = sorted(
        glob.glob(search_pattern),
        key=lambda path: extract_layer_number(os.path.basename(path)),
    )

    if not image_files:
        return html.P(f"No images found in directory: '{server_fs_path}'")

    gallery_content = [html.H3(f"{hover_label}", style={"textAlign": "center"})]
    for img_path_abs in image_files:
        img_filename = os.path.basename(img_path_abs)
        img_src_url = os.path.join(browser_url_path, img_filename)

        layer_name = img_filename.split(".")[0].replace("layer", "tier").replace("_", " ").title()
        component = html.Div(
            [
                html.H5(layer_name),
                html.Img(
                    src=img_src_url,
                    style={
                        "maxWidth": "100%",
                        "border": "1px solid lightgrey",
                        "margin-bottom": "10px",
                    },
                ),
            ]
        )
        gallery_content.append(component)

    return gallery_content


# 6. Run the app locally
if __name__ == "__main__":
    app.run(debug=True)
