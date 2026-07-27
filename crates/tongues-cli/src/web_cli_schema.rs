use std::collections::BTreeSet;

use clap::{Arg, ArgAction, Command, ValueHint};
use serde::Serialize;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct WebCliSchema {
    pub schema_version: u32,
    pub program: String,
    pub commands: Vec<WebCliCommand>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct WebCliCommand {
    /// Stable capability ID, for example `sentence-boundary/train`.
    pub id: String,
    pub name: String,
    pub command: Vec<String>,
    pub route: String,
    pub aliases: Vec<String>,
    pub help: String,
    pub exposed: bool,
    /// Presentation tier for the browser without duplicating command semantics
    /// in JavaScript.
    pub presentation: WebCliPresentation,
    /// Stable browser documentation link for this command.
    pub capability_href: String,
    /// Stable model/capability inventory link when the command uses models.
    pub model_href: Option<String>,
    /// A meaningful Speech Studio starter, when this command has a graph form.
    pub studio_template: Option<String>,
    pub arguments: Vec<WebCliArgument>,
    pub subcommands: Vec<WebCliCommand>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum WebCliPresentation {
    Workflow,
    Component,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct WebCliArgument {
    /// Clap's stable argument ID.
    pub id: String,
    /// Browser/CLI spelling (`--model`) or positional display name (`text`).
    pub name: String,
    pub aliases: Vec<String>,
    pub help: String,
    pub kind: WebCliArgumentKind,
    pub value_type: String,
    pub cardinality: WebCliCardinality,
    pub defaults: Vec<String>,
    pub conflicts: Vec<String>,
    pub required: bool,
    pub global: bool,
    pub value_enum: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum WebCliArgumentKind {
    Flag,
    Option,
    Positional,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct WebCliCardinality {
    pub min: usize,
    pub max: Option<usize>,
    pub repeatable: bool,
}

pub fn build(mut root: Command, exposed_command_ids: &[&str]) -> WebCliSchema {
    // Materialize derive-generated defaults, global arguments, and conflicts.
    root.build();
    let exposed = exposed_command_ids.iter().copied().collect::<BTreeSet<_>>();
    let commands = root
        .get_subcommands()
        .filter_map(|command| command_schema(command, &[], &exposed))
        .collect();
    WebCliSchema {
        schema_version: 1,
        program: root.get_name().to_string(),
        commands,
    }
}

fn command_schema(
    command: &Command,
    parents: &[String],
    exposed: &BTreeSet<&str>,
) -> Option<WebCliCommand> {
    if command.is_hide_set() {
        return None;
    }
    let mut path = parents.to_vec();
    path.push(command.get_name().to_string());
    let id = path.join("/");
    let directly_exposed = exposed.contains(id.as_str());
    let subcommands = command
        .get_subcommands()
        .filter_map(|child| command_schema(child, &path, exposed))
        .collect::<Vec<_>>();
    if !directly_exposed && subcommands.is_empty() {
        return None;
    }

    Some(WebCliCommand {
        id: id.clone(),
        name: command.get_name().to_string(),
        command: path.clone(),
        route: route_for(&path),
        aliases: command.get_all_aliases().map(str::to_string).collect(),
        help: styled(command.get_long_about().or_else(|| command.get_about())),
        exposed: directly_exposed,
        presentation: presentation_for(&id),
        capability_href: format!("/commands/{id}"),
        model_href: model_href_for(&id).map(str::to_string),
        studio_template: studio_template_for(&id).map(str::to_string),
        arguments: command
            .get_arguments()
            .filter(|argument| !argument.is_hide_set())
            .filter(|argument| {
                !matches!(
                    argument.get_action(),
                    ArgAction::Help
                        | ArgAction::HelpShort
                        | ArgAction::HelpLong
                        | ArgAction::Version
                )
            })
            .map(|argument| argument_schema(command, argument))
            .collect(),
        subcommands,
    })
}

fn presentation_for(id: &str) -> WebCliPresentation {
    if matches!(id, "speak" | "speaking" | "predict" | "phonemes" | "phones") {
        WebCliPresentation::Workflow
    } else {
        WebCliPresentation::Component
    }
}

fn model_href_for(id: &str) -> Option<&'static str> {
    (!matches!(
        id,
        "discrepancies" | "fetch-cmudict" | "fetch-corpora" | "phonemes" | "phones"
    ))
    .then_some("/speech/catalog")
}

fn studio_template_for(id: &str) -> Option<&'static str> {
    match id {
        "speak" => Some("text_to_speech"),
        "interpretation/stream" => Some("interpretation"),
        _ => None,
    }
}

fn argument_schema(command: &Command, argument: &Arg) -> WebCliArgument {
    let kind = if argument.is_positional() {
        WebCliArgumentKind::Positional
    } else if matches!(
        argument.get_action(),
        ArgAction::SetTrue | ArgAction::SetFalse | ArgAction::Count
    ) {
        WebCliArgumentKind::Flag
    } else {
        WebCliArgumentKind::Option
    };
    let range = argument.get_num_args();
    let min = range.map_or(0, |range| range.min_values());
    let max = range.and_then(|range| {
        let max = range.max_values();
        (max != usize::MAX).then_some(max)
    });
    let repeatable = matches!(argument.get_action(), ArgAction::Append | ArgAction::Count)
        || max.is_none_or(|max| max > 1);
    let value_enum = argument
        .get_possible_values()
        .into_iter()
        .filter(|value| !value.is_hide_set())
        .map(|value| value.get_name().to_string())
        .collect::<Vec<_>>();
    let value_type = match kind {
        WebCliArgumentKind::Flag => "boolean",
        _ if !value_enum.is_empty() => "enum",
        _ if matches!(
            argument.get_value_hint(),
            ValueHint::AnyPath
                | ValueHint::FilePath
                | ValueHint::DirPath
                | ValueHint::ExecutablePath
        ) =>
        {
            "path"
        }
        _ => "string",
    }
    .to_string();
    let name = argument
        .get_long()
        .map(|long| format!("--{long}"))
        .unwrap_or_else(|| argument.get_id().to_string());

    WebCliArgument {
        id: argument.get_id().to_string(),
        name,
        aliases: argument
            .get_all_aliases()
            .unwrap_or_default()
            .into_iter()
            .map(|alias| format!("--{alias}"))
            .collect(),
        help: styled(argument.get_long_help().or_else(|| argument.get_help())),
        kind,
        value_type,
        cardinality: WebCliCardinality {
            min,
            max,
            repeatable,
        },
        defaults: argument
            .get_default_values()
            .iter()
            .map(|value| value.to_string_lossy().into_owned())
            .collect(),
        conflicts: command
            .get_arg_conflicts_with(argument)
            .into_iter()
            .map(|conflict| conflict.get_id().to_string())
            .collect(),
        required: argument.is_required_set(),
        global: argument.is_global_set(),
        value_enum,
    }
}

fn styled(value: Option<&clap::builder::StyledStr>) -> String {
    value.map(ToString::to_string).unwrap_or_default()
}

fn route_for(path: &[String]) -> String {
    match path {
        [family, command]
            if matches!(
                family.as_str(),
                "g2p2g"
                    | "sentence-boundary"
                    | "head2phones"
                    | "interpretation"
                    | "common-phone"
                    | "emotions"
                    | "wiktionary"
                    | "models"
            ) =>
        {
            format!("/{family}/{command}")
        }
        [family, command] => format!("/cli/{family}/{command}"),
        [command] => format!("/cli/{command}"),
        _ => format!("/cli/{}", path.join("/")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::{Arg, Command};

    #[test]
    fn schema_preserves_clap_shapes_and_filters_unexposed_commands() {
        let command = Command::new("tongues")
            .arg(
                Arg::new("quiet")
                    .long("quiet")
                    .global(true)
                    .action(ArgAction::SetTrue)
                    .conflicts_with("verbose"),
            )
            .arg(
                Arg::new("verbose")
                    .long("verbose")
                    .global(true)
                    .action(ArgAction::SetTrue),
            )
            .subcommand(
                Command::new("family")
                    .alias("f")
                    .about("Current family help")
                    .subcommand(
                        Command::new("run").arg(
                            Arg::new("mode")
                                .long("mode")
                                .value_parser(["safe", "fast"])
                                .default_value("safe")
                                .action(ArgAction::Append),
                        ),
                    )
                    .subcommand(Command::new("destroy")),
            );

        let schema = build(command, &["family/run"]);
        let family = &schema.commands[0];
        assert_eq!(family.aliases, ["f"]);
        assert_eq!(family.help, "Current family help");
        assert_eq!(family.subcommands.len(), 1);
        let run = &family.subcommands[0];
        assert_eq!(run.id, "family/run");
        assert_eq!(run.route, "/cli/family/run");
        assert_eq!(run.capability_href, "/commands/family/run");
        assert_eq!(run.presentation, WebCliPresentation::Component);
        assert_eq!(run.studio_template, None);
        let mode = run.arguments.iter().find(|arg| arg.id == "mode").unwrap();
        assert_eq!(mode.defaults, ["safe"]);
        assert_eq!(mode.value_enum, ["safe", "fast"]);
        assert!(mode.cardinality.repeatable);
        let quiet = run.arguments.iter().find(|arg| arg.id == "quiet").unwrap();
        assert!(quiet.global);
        assert_eq!(quiet.conflicts, ["verbose"]);
    }
}
