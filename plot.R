library(ggplot2)
library(reshape2)
library(dplyr)

fields <- c("max_opt_v", "avg_opt_v")

read_logs <- function(row) {
  filename <- row[1]
  logs <- read.csv(file=filename, header=TRUE, sep=",")
  logs$id <- gsub(".+seed_\\d+_(.+)\\.csv","\\1", filename, perl=TRUE)
  logs$iter <- 1:nrow(logs)
  filtered_logs <- logs[, c("id", "iter", fields)]
  melt(filtered_logs, id=c("id", "iter"))
}

create_run_files <- function(export_dir) {
  files <- c(unlist(sapply(export_dir, function(x) list.files(x, full.names=TRUE, recursive=FALSE))))
  df <- data.frame(file=files)
  df
}

files_df <- create_run_files('./export')
logs <- as.data.frame(do.call(rbind, apply(files_df, 1, read_logs)))
logs$id <- as.factor(logs$id)
logs <- logs[logs$id == 'approx_kernel_ntk_vi',]

facet_names <- c('max_opt_v' = expression("||"~V[N]-tilde(V)^{'*'}~"||"[infinity]), 'avg_opt_v' = expression("||"~V[N]-tilde(V)^{'*'}~"||"[1]))
logs <- mutate_at(logs, .vars = "variable", .funs = factor, labels = facet_names)

logs$variable <- as.factor(logs$variable)

levels(logs$variable)[1] <- expression("||"~V[N]-tilde(V)^{'*'}~"||"[infinity])
levels(logs$variable)[2] <- expression("||"~V[N]-tilde(V)^{'*'}~"||"[1])

summary_logs <- logs %>% group_by(.dots=c('iter', 'variable', 'id')) %>% summarise(ymin=min(value), ymax=max(value), y=mean(value))

beta <- 0.1
K_op <- 0.1941901537963997
gamma_N <- function(N) { (K_op * (beta * 0.9 + 1-beta))^(N-1) * max(logs$value) }

p <- ggplot(summary_logs, aes(x=iter, y=y)) +
  geom_line(aes(colour=id), show.legend = F) +
  geom_function(aes(colour='gamma^N'), fun = gamma_N) +
  geom_ribbon(aes(x=iter, ymax=ymax, ymin=ymin, fill=id), alpha=0.25, linetype=0, show.legend = F) +
  facet_wrap(variable ~., scales='free', ncol=3, labeller=label_parsed) +
  scale_color_discrete(labels = c(expression('Approx Smooth NTK VI'), expression(tilde(gamma)^N))) +
  theme_minimal(base_size=18) +
  theme(strip.text = element_text(size = 20), legend.text.align = 0) +
  labs(x='Iterations', y='', title="", color='')

#show(p)

ggsave("./plot.pdf", p, width=10, height=4)
