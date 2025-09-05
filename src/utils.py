import re
import html


def generate_results_html(path):
    results_dir = path
    image_files = sorted([f for f in results_dir.iterdir() if f.suffix == ".png"])

    # Extract parameters from filenames
    pattern = re.compile(
        r"reward(?P<reward>[\d.]+)_drift(?P<drift>[\d.]+)_noise(?P<noise>[\d.]+)_site(?P<site>[\d.]+)\.png"
    )
    images = []
    for img in image_files:
        m = pattern.match(img.name)
        if m:
            images.append(
                {
                    "filename": img.name,
                    "reward": m.group("reward"),
                    "drift": m.group("drift"),
                    "noise": m.group("noise"),
                    "site": m.group("site"),
                }
            )

    # Get all unique values for each parameter
    rewards = sorted(set(img["reward"] for img in images), key=float)
    drifts = sorted(set(img["drift"] for img in images), key=float)
    noises = sorted(set(img["noise"] for img in images), key=float)
    sites = sorted(set(img["site"] for img in images), key=float)

    # Build product of all settings
    from itertools import product

    param_combinations = list(product(rewards, drifts, noises, sites))

    # Map from param tuple to filename
    param_to_file = {
        (img["reward"], img["drift"], img["noise"], img["site"]): img["filename"]
        for img in images
    }

    # HTML header
    html_parts = [
        "<html><head><title>DDM Results</title><style>table,th,td{border:1px solid #aaa;border-collapse:collapse;} th,td{padding:4px;} img{max-width:600px;display:block;margin:10px 0;} .highlight{background:#ffe;}</style></head><body>"
    ]
    html_parts.append("<h1>DDM Results</h1>")

    # Table of links
    html_parts.append("<h2>Parameter Combinations</h2>")
    html_parts.append(
        "<table><tr><th>Reward</th><th>Drift</th><th>Noise</th><th>Site</th><th>Image</th></tr>"
    )
    for params in param_combinations:
        fname = param_to_file.get(params)
        if fname:
            anchor = fname.replace(".png", "")
            link = f'<a href="#{anchor}">{html.escape(fname)}</a>'
        else:
            link = "<i>Missing</i>"
        html_parts.append(
            f"<tr><td>{params[0]}</td><td>{params[1]}</td><td>{params[2]}</td><td>{params[3]}</td><td>{link}</td></tr>"
        )
    html_parts.append("</table>")

    # Images section
    html_parts.append("<h2>Images</h2>")
    for img_info in images:
        anchor = img_info["filename"].replace(".png", "")
        html_parts.append(
            f'<div id="{anchor}"><h3>{html.escape(img_info["filename"])}</h3><img src="{html.escape(img_info["filename"])}" alt="{html.escape(img_info["filename"])}"></div>'
        )

    html_parts.append("</body></html>")

    out_path = results_dir / "results.html"
    out_path.write_text("\n".join(html_parts), encoding="utf-8")
    print(f"HTML file generated: {out_path.resolve()}")
