#!/bin/bash
# ═══════════════════════════════════════════════
#  ADAS Dataset Generation — Live Monitor
# ═══════════════════════════════════════════════
CSV="/home/hsiraa/adas_ws/src/dataset/adas_dataset.csv"
TOTAL_PER_DRIVER=25
START_TIME=$(date +%s)

while true; do
    clear
    echo "╔══════════════════════════════════════════════════╗"
    echo "║       🚗  ADAS DATASET GENERATION MONITOR       ║"
    echo "╚══════════════════════════════════════════════════╝"

    if [ ! -f "$CSV" ]; then
        echo "  ⏳ Waiting for dataset file..."
        sleep 2
        continue
    fi

    ROWS=$(awk 'END{print NR-1}' "$CSV" 2>/dev/null)
    [ -z "$ROWS" ] && ROWS=0

    # Per-driver counts
    AGG=$(awk -F',' '$12=="AGGRESSIVE"' "$CSV" 2>/dev/null | wc -l)
    NOR=$(awk -F',' '$12=="NORMAL"' "$CSV" 2>/dev/null | wc -l)
    DEF=$(awk -F',' '$12=="DEFENSIVE"' "$CSV" 2>/dev/null | wc -l)

    # Events
    OFF=$(awk -F',' 'NR>1 && $10==1' "$CSV" 2>/dev/null | wc -l)
    COL=$(awk -F',' 'NR>1 && $11==1' "$CSV" 2>/dev/null | wc -l)
    LC=$(awk -F',' 'NR>1 && $13==1' "$CSV" 2>/dev/null | wc -l)
    BRK=$(awk -F',' 'NR>1 && $14==1' "$CSV" 2>/dev/null | wc -l)

    # Current driver + episode estimate
    CURRENT=$(tail -1 "$CSV" 2>/dev/null | cut -d',' -f12)
    CUR_EP=0
    if [ "$CURRENT" = "AGGRESSIVE" ]; then 
        PHASE="1/3"; DR_ROWS=$AGG
        CUR_EP=$(( (DR_ROWS / 2400) + 1 ))
    elif [ "$CURRENT" = "NORMAL" ]; then 
        PHASE="2/3"; DR_ROWS=$NOR
        CUR_EP=$(( (DR_ROWS / 2400) + 1 ))
    elif [ "$CURRENT" = "DEFENSIVE" ]; then 
        PHASE="3/3"; DR_ROWS=$DEF
        CUR_EP=$(( (DR_ROWS / 2400) + 1 ))
    else 
        PHASE="?"; DR_ROWS=0
    fi

    # Elapsed time
    NOW=$(date +%s)
    ELAPSED=$((NOW - START_TIME))
    MINS=$((ELAPSED / 60))
    SECS=$((ELAPSED % 60))

    # Event percentages
    OFF_PCT=0; COL_PCT=0; LC_PCT=0; BRK_PCT=0
    if [ "$ROWS" -gt 0 ]; then
        OFF_PCT=$(echo "scale=1; $OFF*100/$ROWS" | bc 2>/dev/null)
        COL_PCT=$(echo "scale=1; $COL*100/$ROWS" | bc 2>/dev/null)
        LC_PCT=$(echo "scale=1; $LC*100/$ROWS" | bc 2>/dev/null)
        BRK_PCT=$(echo "scale=1; $BRK*100/$ROWS" | bc 2>/dev/null)
    fi

    echo ""
    printf "  🏎  Driver: %-12s  Phase: %s   Episode: %s/25\n" "$CURRENT" "$PHASE" "$CUR_EP"
    printf "  ⏱  Elapsed: %dm %ds\n" "$MINS" "$SECS"
    echo ""
    echo "  ┌─────────────────────────────────────────────┐"
    printf "  │  %-12s  %6d rows                     │\n" "AGGRESSIVE" "$AGG"
    printf "  │  %-12s  %6d rows                     │\n" "NORMAL" "$NOR"
    printf "  │  %-12s  %6d rows                     │\n" "DEFENSIVE" "$DEF"
    echo "  ├─────────────────────────────────────────────┤"
    printf "  │  Total:         %6d rows                  │\n" "$ROWS"
    echo "  └─────────────────────────────────────────────┘"
    echo ""
    echo "  Events:"
    printf "    ⚠  Offroad:    %4d  (%s%%)\n" "$OFF" "$OFF_PCT"
    printf "    💥 Collision:  %4d  (%s%%)\n" "$COL" "$COL_PCT"
    printf "    🔀 Lane Chg:   %4d  (%s%%)\n" "$LC" "$LC_PCT"
    printf "    🛑 Braking:    %4d  (%s%%)\n" "$BRK" "$BRK_PCT"
    echo ""

    # Latest telemetry
    echo "  Latest:"
    tail -1 "$CSV" 2>/dev/null | awk -F',' '{
        printf "    Speed: %5.1f m/s  |  Lat: %+6.2f m  |  Obs: %5.1f m\n", $1, $3, $8
        printf "    Steer: %+6.3f    |  Off: %s  Col: %s  LC: %s\n", $5, $10, $11, $13
    }'
    echo ""
    printf "  Updated: %s\n" "$(date +%H:%M:%S)"
    echo "══════════════════════════════════════════════════"

    sleep 3
done
