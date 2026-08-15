use token_saver::server::run_server;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--mcp") {
        // Run as MCP server over stdio.
        let config = token_saver::config::TokenSaverConfig::load()?;
        token_saver::server::run_mcp_server_with_config(config).await
    } else {
        // Run as HTTP server.
        run_server().await
    }
}
