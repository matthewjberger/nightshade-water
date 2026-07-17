use nightshade::prelude::{App, CameraControllerPlugin, DefaultPlugins, ExitOnEscapePlugin};
use water_app::WaterPlugin;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugin(ExitOnEscapePlugin)
        .add_plugin(CameraControllerPlugin)
        .add_plugin(WaterPlugin)
        .run()
}
