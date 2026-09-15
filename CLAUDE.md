# Previsão Climática — WORCAP/INPE Kaggle Challenge

## Objetivo
Prever precipitação média mensal (mm/dia) sobre América do Sul, 1 mês à frente, dado estado atmosférico do mês atual. Reanálise ERA5/ECMWF, grade 0,25°.

## Domínio
- Grade regular: 301 lat x 261 lon = 78.561 pontos/mês
- Cobertura: 60°S–15°N, 90°O–25°O
- Lat em ordem crescente

## Dados
Treino: jan/1940 – dez/2022. Teste: jan/2023 – dez/2024 (24 meses).
- 2023 → leaderboard público (feedback ao vivo durante competição)
- 2024 → leaderboard privado (classificação final, revelado só no encerramento)
- Precipitação observada dos 24 meses de teste NÃO está em nenhum arquivo distribuído.

### Arquivos de treino (`.nc`, xarray)
- `treino_tp.nc` → var `tp`: precipitação mensal observada, mm/dia, 1940–2022, eixo `time` = mês da observação
- `treino_tp_alvo.nc` → var `tp_alvo`: precipitação do mês SEGUINTE já deslocada (alvo pronto pro treino). Último mês da série = NaN (aponta fora do período)
- 9 outros `treino_<var>.nc`, mesma grade/período, eixo `time` no mês da observação:
  - temperatura a 2m
  - cobertura de nuvens
  - pressão à superfície
  - umidade específica @ 850hPa
  - umidade relativa @ 850hPa
  - temperatura @ 850hPa
  - geopotencial @ 850hPa
  - vento zonal (u_850)
  - vento meridional (v_850)

### Arquivo de teste
`teste_features.nc`:
- eixo `time` = mês ALVO (casa com prefixo do id de submissão)
- campos armazenados = do mês ANTERIOR ao alvo (ver coord `time_origem`)
- `tp_alvo`: inteiramente NaN (é o que precisa ser previsto)
- `tp_ultima_obs`: precipitação de dez/2022 (última observação antes do período avaliado)
- `lag_meses`: defasagem de `tp_ultima_obs` até cada alvo, 1 a 24

### Submissão
`sample_submission.csv` — já traz todos os `id` na ordem certa, `tp_mm_day` zerado. **Não reconstruir o id.**
- Colunas: `id`, `tp_mm_day`
- `id` = `ano_mês_lat_lon` (ex.: `2023_01_-30.00_-53.00`), 2 casas decimais em lat/lon
- Total: 1.885.464 linhas (24 meses x 78.561 pontos)

## Carregamento (padrão)
```python
import xarray as xr

tp = xr.open_dataset("treino_tp.nc").tp               # (time, lat, lon), mm/dia
alvo = xr.open_dataset("treino_tp_alvo.nc").tp_alvo    # precipitação de M+1
vento_u = xr.open_dataset("treino_u_850.nc").u_850
teste = xr.open_dataset("teste_features.nc")

X = tp.values[:-1]      # entrada: mês M
y = alvo.values[:-1]    # saída: mês M+1 (drop last, é NaN)
```

## Avaliação
RMSE (mm/dia) entre previsão e ERA5, sobre todos pontos de grade e todos meses de teste.

**Baseline a bater: climatologia** (média histórica mensal por ponto de grade). Modelo só é bom se superar climatologia de forma consistente, não só reproduzi-la.

## Notas de modelagem
- Problema é forecast espacial + temporal: cada ponto de grade tem série temporal própria, mas há correlação espacial forte (Andes, Amazônia, regime subtropical/temperado no sul do continente).
- Habilidade de modelos numéricos em escala mensal é baixa — não esperar RMSE baixo absoluto, focar em ganho relativo sobre climatologia.
- `lag_meses` no teste varia de 1 a 24 — quanto maior o lag, mais a previsão degrada em direção à climatologia (sem novas observações desde dez/2022 pra atualizar estado).
- Considerar: climatologia mensal por ponto como feature/baseline, features atmosféricas de `treino_*.nc` como preditores, possível vazamento se usar dados futuros.

## Estrutura do projeto
- `data/` — arquivos brutos, já presentes (~1.9 GB):
  - `treino_tp.nc`, `treino_tp_alvo.nc`
  - `treino_t2.nc`, `treino_cloud_cover.nc`, `treino_surface_pressure.nc`
  - `treino_shum_850.nc`, `treino_rel_hum_850.nc`, `treino_temperature_850.nc`
  - `treino_geopotential_850.nc`, `treino_u_850.nc`, `treino_v_850.nc`
  - `teste_features.nc`, `sample_submission.csv`
- `notebooks/` ou `src/` — exploração e pipeline de treino (a criar)
- `submissions/` — CSVs gerados (a criar)

## Atribuição (obrigatória em qualquer publicação)
"Contains modified Copernicus Climate Change Service information 2026. Neither the European Commission nor ECMWF is responsible for any use that may be made of the Copernicus information or data it contains."

## Prazo
Início: 2026-09-14 (aprox). Encerramento: 8 dias a partir do início — **verificar data exata no Kaggle antes de finalizar**.
