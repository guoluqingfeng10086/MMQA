import pandas as pd
import time
from pathlib import Path
from geo_context_loader import (
    get_geologic_epoch,
    get_albedo_value,
    get_mars_elevation_direct,
    get_hirise_context,
    get_paleolake_context,
    get_crater_context,
    get_valley_context,
    get_mineral_abundance
)
# Set path parameters
# Data acquisition can be found in the readme file
albedo_tif_path = r""
elevation_tif_path = r""
paleolake_csv_path = r""
crater_csv_path = r""
crater_csv_path = r""
valley_shp_path = r""
data_dir = Path(r"")
minerals = {
    'Amphibole': data_dir / "TES_Amphibole.tif",
    'Dust': data_dir / "TES_Dust.tif",
    'Hematite': data_dir / "TES_Hematite.tif",
    'High_Si_Glass': data_dir / "TES_High-Silicon_Glass.tif",
    'High_Ca_Px': data_dir / "TES_High_Calcium_Pyroxene.tif",
    'K_Feldspar': data_dir / "TES_K_Feldspar.tif",
    'Low_Ca_Px': data_dir / "TES_Low_Calcium_Pyroxene.tif",
    'Olivine': data_dir / "TES_Olivine.tif",
    'Plagioclase': data_dir / "TES_Plagioclase.tif",
    'Quartz': data_dir / "TES_Quartz.tif"
}
geologic_data_path = r''

def summarize_geological_context(
        epoch=None,
        albedo=None,
        elevation=None,
        hirise_all=None,
        paleolakes=None,
        craters=None,
        valley_groups=None,
        hirise_delta=None,
        mineral_data=None,
        include: list = None,
        **kwargs
) -> str:
    """
    Summarize geomorphological information as multi-line text.
    Use the include parameter to control which parts (field names) are output.
    By default, the focus is on geomorphology-related fields: hirise_all, paleolakes, craters, valley_groups, and mineral_data.
    Fields such as epoch are only included when explicitly listed in include, allowing you to control the output content flexibly.
    """
    if include is None:
        include = ["epoch","albedo","elevation","hirise_all", "paleolakes", "craters", "valley_groups", "mineral_data"]
    lines = []
    if "epoch" in include and epoch:
        lines.append(f"The regional geological epoch is: {epoch}.")
    if "albedo" in include and albedo is not None:
        lines.append(f"The albedo value is approximately {albedo:.3f}.")
    if "elevation" in include and elevation is not None:
        lines.append(f"The elevation is {elevation:.1f} meters.")
    if "hirise_all" in include and hirise_all:
        delta = hirise_delta if hirise_delta is not None else "?"
        lines.append(f"Within ±{delta}° range, {len(hirise_all)} HiRISE terrain images were retrieved:")
        for i, img in enumerate(hirise_all[:30]):
            lines.append(f"  - Image {i + 1}: {img.get('desc', 'Unknown')} ({img.get('url', '')})")
    if "paleolakes" in include and paleolakes:
        lines.append(f"Within ±2° range, {len(paleolakes)} paleolakes were found:")
        for lake in paleolakes:
            lines.append(f"  - Type: {lake.get('Basin Type', 'Unknown')}, Degradation: {lake.get('Degradation', '')}")
    if "craters" in include and craters:
        lines.append(f"{len(craters)} impact craters were found:")
        for crater in craters:
            parts = [f"  - ID: {crater.get('crater_id', '?')}"]
            if pd.notna(crater.get('diameter_km')):
                parts.append(f"Diameter: {crater.get('diameter_km')} km")
            if pd.notna(crater.get('int_morph1')):
                parts.append(f"Interior Morphology: {crater.get('int_morph1')}")
            if pd.notna(crater.get('lay_morph1')):
                parts.append(f"Layered Morphology: {crater.get('lay_morph1')}")
            if pd.notna(crater.get('DEG_RIM')):
                parts.append(f"Rim Degradation: {crater.get('DEG_RIM')}")
            if pd.notna(crater.get('DEG_EJC')):
                parts.append(f"Ejecta Degradation: {crater.get('DEG_EJC')}")
            if pd.notna(crater.get('DEG_FLR')):
                parts.append(f"Floor Degradation: {crater.get('DEG_FLR')}")
            lines.append(', '.join(parts))

    if "valley_groups" in include and valley_groups:
        total_valleys = sum(len(group[1]) for group in valley_groups)
        lines.append(f"A total of {total_valleys} valleys were found:")
        for label, df in valley_groups:
            lines.append(f"  - Range {label}, {len(df)} valleys:")
            for _, row in df.iterrows():
                lines.append(
                    f"    Type: {row.get('Type', '')}, Length: {row.get('Length(km)', '')} km, Epoch: {row.get('age_std', '')}, Distance: {row.get('dist_km', 0):.2f} km"
                )
        lines.append("Note: Higher degradation values indicate better preservation.")

    if "mineral_data" in include and mineral_data:
        lines.append("Estimated mineral abundances at this location:")
        for k, v in mineral_data.items():
            try:
                lines.append(f"  - {k}: {v:.2f}")
            except Exception:
                lines.append(f"  - {k}: {v}")

    return "\n".join(lines)


def query_all_geological_info(lat, lon, features=None):
    """
    Selectively query geological context information based on the parameter.
    features: list, for example ["epoch", "albedo", "elevation"].
    """
    if features is None:
        features = ["epoch", "albedo", "elevation", "hirise", "paleolake", "crater", "valley", "mineral"]

    results = {}
    timings = {}
    if "epoch" in features:
        t0 = time.time()
        results["epoch"] = get_geologic_epoch(lon, lat, geologic_data_path)
        timings["epoch"] = time.time() - t0

    if "albedo" in features:
        t0 = time.time()
        results["albedo"] = get_albedo_value(albedo_tif_path, lon, lat)
        timings["albedo"] = time.time() - t0

    if "elevation" in features:
        t0 = time.time()
        results["elevation"] = get_mars_elevation_direct(elevation_tif_path, lon, lat)
        timings["elevation"] = time.time() - t0

    if "hirise" in features:
        t0 = time.time()
        results["hirise_delta"], results["hirise_all"], results["hirise_top3"] = get_hirise_context(lat, lon)
        timings["hirise"] = time.time() - t0

    if "paleolake" in features:
        t0 = time.time()
        results["paleolakes"] = get_paleolake_context(paleolake_csv_path, lat, lon)
        timings["paleolake"] = time.time() - t0

    if "crater" in features:
        t0 = time.time()
        results["craters"] = get_crater_context(crater_csv_path, lat, lon)
        timings["crater"] = time.time() - t0

    if "valley" in features:
        t0 = time.time()
        results["valley_groups"] = get_valley_context(valley_shp_path, lat, lon, bins_km=[0, 20, 100])
        timings["valley"] = time.time() - t0

    if "mineral" in features:
        t0 = time.time()
        results["mineral_idx"], results["mineral_data"] = get_mineral_abundance(lat, lon, minerals)
        timings["mineral"] = time.time() - t0

    # Printing time information
    print("⏱The time taken for each module is as follows (in seconds):")
    for k, v in timings.items():
        print(f"  {k:<15}: {v:.3f}")
    return results

def format_question_with_context(question: str, context: dict, include: list = None) -> str:
    """
    This appends the specified fields to the question text. `include` specifies the list of fields to include (defaults to `['epoch', 'hirise_top3']`).
    Supported fields (see `query_all_geological_info` return): epoch, hirise_top3, albedo, elevation, ...
    Minimal changes are required: maintain the original logic and output style, but can be controlled by `include`.
    """
    if include is None:
        include = ["epoch", "hirise_top3"]

    result = [question]

    if "epoch" in include and context.get("epoch"):
        result.append(f" The nearby geological epoch is {context['epoch']}")

    if "hirise_top3" in include and context.get("hirise_top3"):
        names = [img.get('desc', 'Unknown') for img in context['hirise_top3']]
        unique_names = []
        seen = set()
        for name in names:
            if name not in seen:
                seen.add(name)
                unique_names.append(name)

        if unique_names:
            result[-1] += f", and nearby HiRISE landforms include {', '.join(unique_names)}"

    if "albedo" in include and context.get("albedo") is not None:
        result.append(f" The albedo is approx {context['albedo']:.3f}.")
    if "elevation" in include and context.get("elevation") is not None:
        result.append(f" Elevation ~{context['elevation']:.1f} m.")
    return "".join(result)

if __name__ == '__main__':
    test_lat = 10.5
    test_lon = 30.2
    lat, lon = 10.5, 30.2
    context = query_all_geological_info(test_lat, test_lon)
    summary = summarize_geological_context(
        epoch=context["epoch"],
        albedo=context["albedo"],
        elevation=context["elevation"],
        hirise_all=context["hirise_all"],
        paleolakes=context["paleolakes"],
        craters=context["craters"],
        valley_groups=context["valley_groups"],
        hirise_delta=context["hirise_delta"],
        mineral_data=context["mineral_data"]
    )
    print(summary)
    example_question = "What is the formation mechanism of Fe-smectite at 25.4°N, 63.2°E?"
    print(format_question_with_context(example_question, context))