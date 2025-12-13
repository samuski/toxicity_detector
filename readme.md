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
In case there were less than 750 rows, custom rows can be provided with `--limit`. At this point it's tail end of the iterations and you must provide this field in later iterations.

```
docker compose exec web \
  python manage.py il_export [iter_count] \
  --limit [row_count]
```

Use this command to check how many rows were human moderated.

```
docker compose exec web python manage.py shell -c "
from main_module.models import ModerationItem;
print(
    ModerationItem.objects.filter(
        decision_source='HUMAN',
        final_action__in=['ALLOW', 'BLOCK'],
    ).count()
)
"
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
    --il_iter 5 \
    --sl_low 0.2 --sl_high 0.8 \
    --il_low 0.2 --il_high 0.8 \
    --il_decision_th 0.5 \
    --review-confident-items
```

## Evaluation

Evaluation requires paths to test csv file and the weights.
`filename` is straightfoward.
`weight_path` is `sl/baseline` or `sl/oracle` for SL, and `il/iter_[num]/artifacts` where num is three digits.

```
docker compose exec web \
  python /app/main_module/sl/eval.py \
    --data_csv /app/data/[filename].csv \
    --label_col label_score \
    --label_threshold 0.5 \
    --weights /artifacts/[weight_path]
```
