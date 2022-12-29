#Librarias
library(stringr)
library(dplyr)
library(readr)
library(tokenizers)
library(tidytext)
library(tidyverse)
library(readxl)
library(tidymodels)
library(textrecipes)
library(discrim)
library(naivebayes)
library(glmnet)
library(reticulate)
library(udpipe)
library(quanteda)
library(lubridate)
library(stm)
library(tm)

#Importar y generar listas de stopwords ----
full_corpus_aborto <- read_xlsx("Aborto_FavCon.xlsx")
corpus_contra <- read_xlsx("Intervención_Contra.xlsx")
corpus_favor <- read_xlsx("Intervención_Favor.xlsx")
stopwords_anatext <- read_csv("https://raw.githubusercontent.com/7PartidasDigital/AnaText/master/datos/diccionarios/vacias.txt")
stopwords_adicionales <- tibble(palabra = c("artículo", "artículos", "ministro", "ministra", "senador", "senadora", "diputado", "diputada", "parlamentario", "parlamentarios", "señor", "señora", "señores", "presidente", "(...)"))
stopwords_full = stopwords_anatext %>% full_join(stopwords_adicionales, by= "palabra")

#Tokenizacion y Lematización ----

#Descargando y cargando modelo de español GSD
udpipe_download_model(language = "spanish-gsd")

modelo_gsd <- udpipe_load_model(file = "spanish-gsd-ud-2.5-191206.udpipe")

#Anotando  sobre los corpus con su respectivos lemas
full_corpus_aborto_anotado <- udpipe_annotate(object = modelo_gsd, x =  full_corpus_aborto$intervencion)
full_corpus_aborto_anotado <- as_tibble(full_corpus_aborto_anotado)

corpus_contra_anotado <- udpipe_annotate(object = modelo_gsd, x =  corpus_contra$intervencion) 
corpus_contra_anotado <- as_tibble(corpus_contra_anotado)

corpus_favor_anotado <- udpipe_annotate(object = modelo_gsd, x =  corpus_favor$intervencion)
corpus_favor_anotado <- as_tibble(corpus_favor_anotado)

# Contando lemas
frecuencia_lema_full <- full_corpus_aborto_anotado %>% 
  filter(upos %in% c("NOUN", "VERB", "ADJ", "ADV", "PROPN")) %>%
  count(lemma, sort = TRUE) %>% 
  anti_join(stopwords_full, by= c("lemma" = "palabra"))

frecuencia_lema_contra <- corpus_contra_anotado %>% 
  filter(upos %in% c("NOUN", "VERB", "ADJ", "ADV", "PROPN")) %>%
  count(lemma, sort = TRUE) %>% 
  anti_join(stopwords_full, by= c("lemma" = "palabra"))

frecuencia_lema_favor <- corpus_favor_anotado %>% 
  filter(upos %in% c("NOUN", "VERB", "ADJ", "ADV", "PROPN")) %>%
  count(lemma, sort = TRUE) %>% 
  anti_join(stopwords_full, by= c("lemma" = "palabra"))

frecuencia_lema_resto <- corpus_favor_anotado %>% 
  filter(upos %in% c("NOUN", "VERB", "ADJ", "ADV", "PROPN")) %>%
  count(lemma, sort = TRUE) %>% 
  anti_join(stopwords_full, by= c("lemma" = "palabra")) %>% 
  anti_join(frecuencia_lema_contra, by = "lemma")

#Tokenizar por texto del corpus general
tokenize_words(full_corpus_aborto$intervencion, 
               strip_punct = TRUE,
               strip_numeric = TRUE,
               stopwords = stopwords_full$palabra)

#Sacar 100 tokens más frecuentes del corpus general
tibble(discusion_general = full_corpus_aborto$intervencion) %>% 
  unnest_tokens(input = discusion_general, output = palabra) %>% 
  count(palabra, sort = TRUE) %>% 
  anti_join(stopwords_full)

#Sacar 100 tokens más frecuentes del corpus a favor
tibble(discusion_general = corpus_favor$intervencion) %>% 
  unnest_tokens(input = discusion_general, output = palabra) %>% 
  count(palabra, sort = TRUE) %>% 
  anti_join(stopwords_full) %>% 
  print(n=20)

#Sacar 100 tokens más frecuentes del corpus a favor
tibble(discusion_general = corpus_contra$intervencion) %>% 
  unnest_tokens(input = discusion_general, output = palabra) %>% 
  count(palabra, sort = TRUE) %>% 
  anti_join(stopwords_full) %>% 
  print(n=100)

#TF-IDF ----

#Frecuencia votos en contra
frecuencia_contra <- tibble(discurso = corpus_contra$intervencion) %>% 
  unnest_tokens(input = discurso,
                output = palabra,
                strip_numeric = TRUE) %>% 
  count(palabra, sort = TRUE) %>%
  anti_join(stopwords_full) %>% 
  print(n=100)

frecuencia_contra <- mutate(frecuencia_contra, voto = "contra", .before = palabra)

#Frecuencia votos a favor
frecuencia_favor <- tibble(discurso = corpus_favor$intervencion) %>% 
  unnest_tokens(input = discurso,
                output = palabra,
                strip_numeric = TRUE) %>%
  count(palabra, sort = TRUE) %>%
  anti_join(stopwords_full) %>% 
  print(n=100)

frecuencia_favor <- mutate(frecuencia_favor, voto = "favor", .before = palabra)


#Union de ambos discursos
frecuencia_totales <- bind_rows(frecuencia_contra, frecuencia_favor)

#TF-IDF más alto por tipo de voto
tfidf_full <- bind_tf_idf(frecuencia_totales,
                                  term = palabra,
                                  document = voto,
                                  n=n)

tfidf_full %>%  filter(voto == "contra") %>% 
  slice_max(tf_idf, n=20) %>% 
  View()

tfidf_full %>%  filter(voto == "favor") %>% 
  slice_max(tf_idf, n=20) %>% 
  View()

#Comparando (proceso repetido de otra forma?)
tfidf_votos <- frecuencia_totales %>% 
  group_by(voto, palabra) %>% 
  summarise(n = sum(n)) %>% 
  ungroup() %>% 
  bind_tf_idf(term = palabra,
              document = voto,
              n = n) 

#Modelo de clasificacion Naive Bayes----
head(full_corpus_aborto$intervencion)

#Limpiar numeros
full_corpus_aborto %>% 
  mutate(intervencion = str_remove_all(intervencion, "\\b[0-9]*")) %>%
  head() %>% 
  view()

#Construir modelo y hacer split del corpus total
set.seed(1234)

aborto_total_split <- initial_split(full_corpus_aborto, strata = voto)

#Separar conjunto de prueba y entrenamiento
aborto_train <- training(aborto_total_split)
dim(aborto_train)

aborto_test <- testing(aborto_total_split)
dim(aborto_test)

#Una vez separamos conjuntos, pre procesamos la data
aborto_rec <-
  recipe(voto ~ intervencion, data = aborto_train)

#Tokenizamos  y sacamos tf idf como parte de la receta mediante Tokenizers
#Falta agregar lematización
aborto_rec <- aborto_rec %>%
  step_tokenize(intervencion, options = list(strip_numeric = TRUE)) %>%
  step_tokenfilter(intervencion, max_tokens = 1e3) %>%
  step_tfidf(intervencion)

#Aplicamos la receta
aborto_wf <- workflow() %>% 
  add_recipe(aborto_rec)

#Llamamos modelo Naive Bayes
nb_spec <- naive_Bayes() %>%
  set_mode("classification") %>%
  set_engine("naivebayes")

nb_spec

#Entrenamos el modelo de clasificacion
nb_fit <- aborto_wf %>%
  add_model(nb_spec) %>%
  fit(data = aborto_train)

#K-fold evaluation en 10
set.seed(234)
aborto_folds <- vfold_cv(aborto_train)

aborto_folds

nb_wf <- workflow() %>%
  add_recipe(aborto_rec) %>%
  add_model(nb_spec)

nb_wf

#Insertar data resamples
nb_rs <- fit_resamples(
  nb_wf,
  aborto_folds,
  control = control_resamples(save_pred = TRUE))

#Colectando metricas
nb_rs_metrics <- collect_metrics(nb_rs)
nb_rs_predictions <- collect_predictions(nb_rs)

#Matriz de confusion
conf_mat_resampled(nb_rs, tidy = FALSE) %>%
  autoplot(type = "heatmap")

#Modelo de clasificación Lasso----
#Llamamos al model tuning para aplicar Lasso
tune_spec <- logistic_reg(penalty = tune(), mixture = 1) %>%
  set_mode("classification") %>%
  set_engine("glmnet")

tune_spec

#Creamos una matriz de valores para probar la conveniencia para penalty
lambda_grid <- grid_regular(penalty(), levels = 30)
lambda_grid

#Aplicamos el tuning al workflow
tune_wf <- workflow() %>%
  add_recipe(aborto_rec) %>%
  add_model(tune_spec)

#Generamos los Folds
set.seed(2020)

tune_rs <- tune_grid(tune_wf,
                     aborto_folds,
                     grid = lambda_grid,
                     control = control_resamples(save_pred = TRUE))
tune_rs

#Sacamos métricas y visualizamos
collect_metrics(tune_rs)

autoplot(tune_rs)
 
#Seleccionamos el mejor penalty pensando en ROC
chosen_auc <- tune_rs %>%
  select_by_one_std_err(metric = "roc_auc", -penalty)

chosen_auc

#Y cerramos el workflow de Lasso
final_lasso <- finalize_workflow(tune_wf, chosen_auc)

final_lasso

#ajustamos K-Fold con el penalty
fitted_lasso <- fit(final_lasso, aborto_train)

fitted_lasso %>%
  extract_fit_parsnip() %>%
  tidy() %>%
  arrange(-estimate)



#Evaluacion----
#Colectando metricas
nb_rs_metrics <- collect_metrics(nb_rs)
nb_rs_predictions <- collect_predictions(nb_rs)

nb_rs_metrics

nb_rs_predictions %>% 
  recall(voto, .pred_class)

nb_rs_predictions %>% 
  precision(voto, .pred_class)
  p

#Sacando el ROC por fold
nb_rs_predictions %>%
  group_by(id) %>%
  roc_curve(truth = voto, .pred_class) %>%
  autoplot() +
  labs(
    color = NULL,
    title = "Curva ROC para la predicción de voto")

#Topic Model (STM)----
#importar datos
full_corpus_aborto <- read_xlsx("Aborto_FavCon.xlsx")
stopwords_anatext <- read_csv("https://raw.githubusercontent.com/7PartidasDigital/AnaText/master/datos/diccionarios/vacias.txt")
stopwords_adicionales <- tibble(palabra = c("artículo", "artículos", "ministro", "ministra", "senador", "senadora", "diputado", "diputada", "parlamentario", "parlamentarios", "señor", "señora", "señores", "presidente", "(...)"))
stopwords_full = stopwords_anatext %>% full_join(stopwords_adicionales, by= "palabra")

#Dar el formato correcto
full_corpus_aborto
procesado <- textProcessor(full_corpus_aborto$intervencion,
                           language = "es",
                           removepunctuation = TRUE,
                           removenumbers = TRUE,
                           lowercase = TRUE,
                           customstopwords = stopwords_full$palabra,
                           metadata = full_corpus_aborto)

out <- prepDocuments(procesado$documents,
                     procesado$vocab,
                     procesado$meta)

#sacando terminos infrecuentes
plotRemoved(procesado$documents,
            lower.thresh = seq(1, 200, by=100))

out <- prepDocuments(procesado$documents,
                     procesado$vocab,
                     procesado$meta,
                     lower.thresh = 15)                            

#Estimando
abortoFit <- stm(documents = out$documents,
                 vocab = out$vocab,
                 K = 4,
                 content = out$meta$voto,
                 max.em.its = 75,
                 data = out$meta,
                 init.type = "Spectral")

#seleccionando numero de topicos
storage <- searchK(out$documents,
                   out$vocab,
                   init.type = "Spectral", 
                   M = 10,
                   K = c(3, 4, 5, 6, 7, 8, 9, 10),
                   data = meta)

plot(storage)

#seleccionando el modelo
abortoSelect <- selectModel(out$documents,
                            out$vocab,
                            K=4,
                            content = out$meta$voto,
                            max.em.its = 75,
                            data = out$meta,
                            runs = 20,
                            seed = 8458159)

plotModels(abortoSelect,
           pch = c(1, 2, 3, 4),
           legend.position = "bottomright")

selectedmodel <- abortoSelect$runout[[2]]

#Preguntar a Riva porqué no funciona
topicQuality(model = selectedmodel,
             documents = out$documents,
             xlab = "Semantic Coherence",
             ylab = "Exclusivity",
             labels = 1:ncol(selectedmodel$theta),
             M = 10)

#visualización de palabras asociadas al topico
labelTopics(selectedmodel)

#visualización de porción de corpus utilizada por cada tema
plot(selectedmodel, type = "summary", xlim = c(0, 1))

