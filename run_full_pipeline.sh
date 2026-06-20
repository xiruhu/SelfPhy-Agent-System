#!/bin/bash
# SelfPhy-Agent-System 完整 Pipeline
#
# 用法:
#   bash run_full_pipeline.sh               # 自动生成 RUN_ID（时间戳）
#   RUN_ID=v2_kimi_doubao bash run_full_pipeline.sh  # 指定 RUN_ID
#
# 输出目录结构:
#   outputs/runs/<RUN_ID>/exams/
#   outputs/runs/<RUN_ID>/answers/
#   outputs/runs/<RUN_ID>/reports/
#   outputs/runs/<RUN_ID>/visualizations/
#
# Resume 逻辑：检查输出文件是否存在，存在则 skip。
# 注意：pipeline 不感知代码变化。如需重跑某步，手动删除对应输出文件。

PROJECT=/mnt/cpfs/yqh/HXR/hxr_SelfPhy-Agent-System
TRAJ_DATA=$PROJECT/data/VL-LN-Bench/traj_data/mp3d_split2
EXTRACTED=$PROJECT/data/VL-LN-Bench/extracted

RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
RUN_DIR=$PROJECT/outputs/runs/$RUN_ID
LOG=$RUN_DIR/pipeline.log

MODELS="kimi doubao"
MAX_EPISODES=1
NUM_QUESTIONS=5

SCENES="D7N2EKCX4Sj 17DRP5sb8fy 1LXtFkjw3qL 1pXnuDYAj8r 29hnd4uzFmX 2n8kARJN3HM 5LpN3gDmAk7 5q7pvUzZiYa 759xd9YjKW5 7y3sRwLe3Va"

mkdir -p $RUN_DIR/exams $RUN_DIR/answers $RUN_DIR/reports $RUN_DIR/visualizations $EXTRACTED

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

log "======================================"
log "SelfPhy-Agent-System 完整 Pipeline"
log "RUN_ID: $RUN_ID"
log "RUN_DIR: $RUN_DIR"
log "MODELS: $MODELS"
log "======================================"

# ─── Step 1: 从每个场景提取 1 个 episode ───
log ""
log "[Step 1] 从各场景 tar.gz 提取 episode..."
cd $PROJECT/tools
for SCENE in $SCENES; do
    TAR=$TRAJ_DATA/${SCENE}.tar.gz
    OUT=$EXTRACTED/$SCENE
    DONE_MARKER=$OUT/.extracted

    if [ -f "$DONE_MARKER" ]; then
        log "  [SKIP] $SCENE: 已提取"
        continue
    fi

    FSIZE=$(stat -c%s "$TAR" 2>/dev/null || echo 0)
    if [ ! -f "$TAR" ] || [ "$FSIZE" -lt 10000 ]; then
        log "  [SKIP] $SCENE: tar.gz 不存在或为空 (${FSIZE}B)"
        continue
    fi

    log "  [PROCESS] $SCENE ($(du -sh $TAR | cut -f1))"
    python3 extract_all_episodes.py \
        --tar $TAR \
        --output $OUT \
        --max-episodes $MAX_EPISODES && touch $DONE_MARKER
done
log "[Step 1] 完成"

# ─── Step 2: 生成 metadata ───
log ""
log "[Step 2] 生成 metadata.json..."
cd $PROJECT
for SCENE in $SCENES; do
    for ep_dir in $EXTRACTED/$SCENE/episode_*/; do
        [ -d "$ep_dir" ] || continue
        ep_name=$(basename $ep_dir)
        traj_file=$ep_dir/trajectory.json
        meta_file=$ep_dir/metadata.json

        [ -f "$traj_file" ] || { log "  [SKIP] $SCENE/$ep_name: 无 trajectory.json"; continue; }
        [ -f "$meta_file" ] && { log "  [SKIP] $SCENE/$ep_name: metadata 已存在"; continue; }

        log "  [PROCESS] $SCENE/$ep_name"
        python3 core/habitat_metadata_builder.py $traj_file -o $meta_file
    done
done
log "[Step 2] 完成"

# ─── Step 2.5: 复用已有考卷（EXAM_SOURCE_RUN） ───
if [ -n "$EXAM_SOURCE_RUN" ]; then
    SRC_EXAM_DIR=$PROJECT/outputs/runs/$EXAM_SOURCE_RUN/exams
    log ""
    log "[Step 2.5] 从 $EXAM_SOURCE_RUN 复制考卷（跳过 Claude 出题）..."
    if [ -d "$SRC_EXAM_DIR" ]; then
        count=0
        for f in $SRC_EXAM_DIR/exam_*.json; do
            [ -f "$f" ] || continue
            base=$(basename "$f")
            # 只复制原始考卷（非 refined），不覆盖已存在的
            [[ "$base" == exam_refined_* ]] && continue
            dest=$RUN_DIR/exams/$base
            if [ -f "$dest" ]; then
                log "  [SKIP] $base: 已存在"
            else
                cp "$f" "$dest"
                count=$((count+1))
                log "  [COPY] $base"
            fi
        done
        log "[Step 2.5] 完成：复制 $count 份考卷"
    else
        log "[Step 2.5] WARN: 源目录不存在 ($SRC_EXAM_DIR)，跳过复制"
    fi
fi

# ─── Step 3: 生成考题 ───
log ""
log "[Step 3] 生成考题（Claude Examiner，每 episode $NUM_QUESTIONS 题）..."
for SCENE in $SCENES; do
    for ep_dir in $EXTRACTED/$SCENE/episode_*/; do
        [ -d "$ep_dir" ] || continue
        ep_name=$(basename $ep_dir)
        meta_file=$ep_dir/metadata.json
        exam_file=$RUN_DIR/exams/exam_${SCENE}_${ep_name}.json

        [ -f "$meta_file" ] || { log "  [SKIP] $SCENE/$ep_name: 无 metadata"; continue; }
        [ -f "$exam_file" ] && { log "  [SKIP] $SCENE/$ep_name: 考题已存在"; continue; }

        log "  [PROCESS] $SCENE/$ep_name"
        python3 core/claude_examiner.py $meta_file \
            --data-dir $ep_dir \
            -n $NUM_QUESTIONS \
            -o $exam_file
    done
done
log "[Step 3] 完成"

# ─── Step 3.5: GPT-4o-mini 质量审核 ───
log ""
log "[Step 3.5] GPT-4o 质量审核考卷（验算答案 + 判断空间推理有效性）..."
for SCENE in $SCENES; do
    for ep_dir in $EXTRACTED/$SCENE/episode_*/; do
        [ -d "$ep_dir" ] || continue
        ep_name=$(basename $ep_dir)
        meta_file=$ep_dir/metadata.json
        exam_file=$RUN_DIR/exams/exam_${SCENE}_${ep_name}.json
        refined_file=$RUN_DIR/exams/exam_refined_${SCENE}_${ep_name}.json

        [ -f "$exam_file" ] || { log "  [SKIP] $SCENE/$ep_name: 无原始考卷"; continue; }
        [ -f "$refined_file" ] && { log "  [SKIP] $SCENE/$ep_name: 精炼考卷已存在"; continue; }

        log "  [PROCESS] $SCENE/$ep_name"
        python3 core/gpt_quality_checker.py $exam_file $meta_file \
            -o $refined_file || {
            log "  [WARN] $SCENE/$ep_name 质量审核失败，将原始考卷复制为精炼考卷"
            cp $exam_file $refined_file
        }
    done
done
log "[Step 3.5] 完成"

# ─── Step 4: 被测模型评测 ───
log ""
log "[Step 4] 被测模型评测（$MODELS，使用精炼考卷）..."
for SCENE in $SCENES; do
    for ep_dir in $EXTRACTED/$SCENE/episode_*/; do
        [ -d "$ep_dir" ] || continue
        ep_name=$(basename $ep_dir)
        # 优先使用精炼考卷，降级使用原始考卷
        refined_file=$RUN_DIR/exams/exam_refined_${SCENE}_${ep_name}.json
        orig_file=$RUN_DIR/exams/exam_${SCENE}_${ep_name}.json
        if [ -f "$refined_file" ]; then
            exam_file=$refined_file
        elif [ -f "$orig_file" ]; then
            exam_file=$orig_file
            log "  [WARN] $SCENE/$ep_name: 使用原始考卷（精炼版不存在）"
        else
            continue
        fi

        for model in $MODELS; do
            result_file=$RUN_DIR/answers/result_${model}_${SCENE}_${ep_name}.json
            [ -f "$result_file" ] && { log "  [SKIP] $SCENE/$ep_name $model: 已评测"; continue; }

            log "  [PROCESS] $SCENE/$ep_name → $model"
            python3 core/evaluate_runner.py $exam_file \
                --model $model \
                -o $result_file || log "  [WARN] $SCENE/$ep_name $model 失败，继续..."
        done
    done
done
log "[Step 4] 完成"

# ─── Step 5: 错题诊断 ───
log ""
log "[Step 5] 错题诊断（Claude Reflector）..."
for SCENE in $SCENES; do
    for ep_dir in $EXTRACTED/$SCENE/episode_*/; do
        [ -d "$ep_dir" ] || continue
        ep_name=$(basename $ep_dir)
        # 诊断优先使用精炼考卷（答案已更正）
        refined_exam=$RUN_DIR/exams/exam_refined_${SCENE}_${ep_name}.json
        orig_exam=$RUN_DIR/exams/exam_${SCENE}_${ep_name}.json
        if [ -f "$refined_exam" ]; then
            exam_file=$refined_exam
        elif [ -f "$orig_exam" ]; then
            exam_file=$orig_exam
        else
            continue
        fi
        meta_file=$ep_dir/metadata.json

        [ -f "$exam_file" ] || continue

        for model in $MODELS; do
            result_file=$RUN_DIR/answers/result_${model}_${SCENE}_${ep_name}.json
            diag_file=$RUN_DIR/reports/diagnosis_${model}_${SCENE}_${ep_name}.json

            [ -f "$result_file" ] || continue
            [ -f "$diag_file" ] && { log "  [SKIP] $SCENE/$ep_name $model: 诊断已存在"; continue; }

            log "  [PROCESS] $SCENE/$ep_name $model"
            python3 core/claude_reflector.py $result_file $exam_file $meta_file \
                -o $diag_file || log "  [WARN] $SCENE/$ep_name $model 诊断失败，继续..."
        done
    done
done
log "[Step 5] 完成"

# ─── Step 6: 可视化 ───
log ""
log "[Step 6] 生成可视化报告..."
cd $PROJECT

RESULTS_ARGS=""
DIAG_ARGS=""
for model in $MODELS; do
    for f in $RUN_DIR/answers/result_${model}_*.json; do
        [ -f "$f" ] && RESULTS_ARGS="$RESULTS_ARGS ${model}:$f"
    done
    for f in $RUN_DIR/reports/diagnosis_${model}_*.json; do
        [ -f "$f" ] && DIAG_ARGS="$DIAG_ARGS ${model}:$f"
    done
done

if [ -n "$RESULTS_ARGS" ]; then
    python3 core/analytics_viz.py \
        --results $RESULTS_ARGS \
        --diagnoses $DIAG_ARGS \
        --output-dir $RUN_DIR/visualizations || log "[WARN] 可视化失败"
fi
log "[Step 6] 完成"

log ""
log "======================================"
log "全部完成！"
log "RUN_ID: $RUN_ID"
log "考题: $(ls $RUN_DIR/exams/*.json 2>/dev/null | wc -l) 份"
log "答案: $(ls $RUN_DIR/answers/*.json 2>/dev/null | wc -l) 份"
log "诊断: $(ls $RUN_DIR/reports/*.json 2>/dev/null | wc -l) 份"
log "日志: $LOG"
log ""
log "查看结果: streamlit run streamlit_dashboard.py"
log "  (如需通过 SSH 访问: ssh -N -L 8501:localhost:8501 <server>)"
log "======================================"
