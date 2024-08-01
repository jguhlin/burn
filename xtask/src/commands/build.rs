use xtask_common::{
    anyhow,
    commands::build::{self, BuildCmdArgs},
    utils::helpers,
    ExecutionEnvironment,
};

use crate::{ARM_TARGET, NO_STD_CRATES, WASM32_TARGET};

pub(crate) fn handle_command(
    mut args: BuildCmdArgs,
    exec_env: ExecutionEnvironment,
) -> anyhow::Result<()> {
    match exec_env {
        ExecutionEnvironment::NoStd => {
            ["Default", WASM32_TARGET, ARM_TARGET]
                .iter()
                .try_for_each(|build_target| {
                    let mut build_args = vec!["--no-default-features"];
                    if *build_target != "Default" {
                        build_args.extend(vec!["--target", *build_target]);
                    }
                    helpers::custom_crates_build(NO_STD_CRATES.to_vec(), build_args)
                })?;
            Ok(())
        }
        ExecutionEnvironment::Std => {
            // Exclude crates that are not supported on CI
            args.exclude
                .extend(vec!["burn-cuda".to_string(), "burn-tch".to_string()]);
            if std::env::var("DISABLE_WGPU").is_ok() {
                args.exclude.extend(vec!["burn-wgpu".to_string()]);
            };
            // Build workspace
            build::handle_command(args.clone())?;
            // Specific additional commands to test specific features
            // burn-dataset
            helpers::custom_crates_build(vec!["burn-dataset"], vec!["--all-features"])?;
            Ok(())
        }
    }
}
