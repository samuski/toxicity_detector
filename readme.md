# Commands

## IL

### Ingest

augment.csv into DB and run SL baseline inference for uncertain items.

```
  docker compose exec web \
  python manage.py il_ingest \
  /app/data/augment.csv \
  --source augment \
  --label-col label_score \
  --low 0.2 \
  --high 0.8
```

### Export

Export the human moderated items. Default is 750 and need to provide iteration number.

```
docker compose exec web \
  python manage.py il_export [iter_count]
```

### Train

Trains the IL iteration. Need to provide iteration number.

```
docker compose exec web \
  python manage.py il_train [iter_count]
```

### Scan

Use new model to check for SL and IL disagreement and convert uncertain items to certain if it's very certain.
[Optional] `--review-confident-items` This tag enables review of all the items that was marked as confident to have new iteration review the old

```
docker compose exec web \
  python manage.py il_scan \
    --source augment \
    --il_iter [iter_count] \
    --sl_low 0.2 --sl_high 0.8 \
    --il_low 0.2 --il_high 0.8 \
    --il_decision_th 0.5 \
    --review-confident-items
```
