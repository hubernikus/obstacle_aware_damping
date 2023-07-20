from attrs import define, field


@define
class PlotStyle:
    color: str
    secondary_color: str
    label: str


plot_setup = {
    "dynamics": PlotStyle(
        color="#AD0500",
        secondary_color="#F56662",
        label="Dynamics preserving",
    ),
    "obstacle": PlotStyle(
        color="#005FA8",
        secondary_color="#70BEFA",
        label="Obstacle aware",  # "Dynamcis"
    ),
    "active_color": "#A9AB30",
}
