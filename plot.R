library(ggplot2)
library(reshape2)
library(dplyr)

setwd('C:/Projects/deep-rl/dp/export/')

logs <- read.csv(file='errors.csv', header=TRUE, sep=",")
logs$iter <- 1:nrow(logs)

melted_logs <- melt(logs, id=c("iter"))

p <- ggplot(melted_logs, aes(x=iter, y=value, group=variable, colour=variable)) +
  geom_line() + 
  facet_wrap(variable ~., scales='free', ncol=3) +
  theme_minimal() +
  theme(legend.title=element_blank()) +
  labs(x='Iterations', y='', title="")

show(p)


fields <- c("max_opt_v", "avg_opt_v")
labels <- c("||V_k-V^*||_\inf", "||V_k-V^*||_1")
fields_labels = list()
for (i in seq_along(fields)) {
    fields_labels[[fields[i]]] = labels[i]
}                       

myLabeller <- function(x){
  lapply(x,function(y){
    y_char <- as.character(y)
    if (y %in% fields) {
      fields_labels[[y_char]]
    } else {
      y
    }
  })
}

read_logs <- function(row) {
  filename <- row[1]
  logs <- read.csv(file=filename, header=TRUE, sep=",")
  logs$id <- paste0("", logs$tau)
  logs[logs$id == 0, c("id")] <- 1 # is_soft=False implies tau = 1 
  logs$iter <- 1:nrow(logs)
  filtered_logs <- logs[, c("id", "iter", fields)]
  melt(filtered_logs, id=c("id", "iter"))
}

create_run_files <- function(export_dir) {
  files <- c(unlist(sapply(export_dir, function(x) list.files(x, full.names=TRUE, recursive=FALSE))))
  df <- data.frame(file=files)
  df
}

files_df <- create_run_files('C:/Projects/deep-rl/dp/export_beta')
logs <- as.data.frame(do.call(rbind, apply(files_df, 1, read_logs)))

facet_names <- c('max_opt_v' = expression("||"~V[k]-V^{'*'}~"||"[infinity]), 'avg_opt_v' = expression("||"~V[k]-V^{'*'}~"||"[1]))
logs <- mutate_at(logs, .vars = "variable", .funs = factor, labels = facet_names)

logs$variable <- as.factor(logs$variable)

levels(logs$variable)[1] <- expression("||"~V[k]-V^{'*'}~"||"[infinity])
levels(logs$variable)[2] <- expression("||"~V[k]-V^{'*'}~"||"[1])

summary_logs <- logs %>% group_by(.dots=c('iter', 'variable', 'id')) %>% summarise(ymin=min(value), ymax=max(value), y=mean(value))

p <- ggplot(summary_logs, aes(x=iter, y=y)) +
  #scale_fill_hue(expression(beta)) +
  geom_line(aes(colour=id), show.legend = F) + 
  geom_ribbon(aes(x=iter, ymax=ymax, ymin=ymin, fill=id), alpha=0.25, linetype=0) +
  facet_wrap(variable ~., scales='free', ncol=3, labeller=label_parsed) +
  theme_minimal(base_size=18) +
  theme(strip.text = element_text(size = 20)) +
  labs(x='Iterations', y='', title="") + 
  scale_color_manual(values=c("#000000", "#E69F00", "#56B4E9", "#009E73",
                              "#F0E442", "#0072B2", "#D55E00", "#CC79A7")) + 
  scale_fill_manual(expression(beta), values=c("#000000", "#E69F00", "#56B4E9", "#009E73",
                              "#F0E442", "#0072B2", "#D55E00", "#CC79A7"))

show(p)

ggsave("plot_beta.pdf", p, width=10, height=4)
