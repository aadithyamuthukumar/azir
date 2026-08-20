package handlers

import (
	"context"
	"encoding/json"
	"net/http"

	"github.com/aadithyamuthukumar/azir/models"
	"github.com/jackc/pgx/v5"
)

func Optimizations(w http.ResponseWriter, r *http.Request, conn *pgx.Conn) {
	var summaries []models.OptimizationSummary
	rows, err := conn.Query(
		context.Background(),
		`SELECT
			feature,
			AVG(input_tokens),
			AVG(output_tokens),
			COUNT(*),
			SUM(total_cost)
		FROM llm_events
		GROUP BY feature
		HAVING AVG(input_tokens) > 2000 OR AVG(output_tokens) > 2000
		ORDER BY AVG(input_tokens) DESC, AVG(output_tokens) DESC`)

	if err != nil {
		http.Error(w, "failed to fetch optimizations", http.StatusInternalServerError)
		return
	}
	defer rows.Close()

	for rows.Next() {
		var summary models.OptimizationSummary
		err := rows.Scan(
			&summary.Feature,
			&summary.AvgInputTokens,
			&summary.AvgOutputTokens,
			&summary.Requests,
			&summary.TotalCost,
		)
		if err != nil {
			http.Error(w, "failed to scan optimization", http.StatusInternalServerError)
			return
		}

		if summary.AvgInputTokens > 2000 {
			summary.Type = "large_input"
			summary.Message = "This feature has unusually large input contexts."
		} else if summary.AvgOutputTokens > 2000 {
			summary.Type = "large_output"
			summary.Message = "This feature has unusually large output contexts."
		}
		summaries = append(summaries, summary)
	}
	if err := rows.Err(); err != nil {
		http.Error(w, "failed to read optimizations", http.StatusInternalServerError)
		return
	}

	repeatedRows, err := conn.Query(
		context.Background(),
		`SELECT
			feature,
			prompt_hash,
			COUNT(*) AS requests,
			AVG(input_tokens) AS avg_input_tokens,
			SUM(input_cost) AS total_input_cost
		FROM llm_events
		WHERE prompt_hash IS NOT NULL
		GROUP BY feature, prompt_hash
		HAVING COUNT(*) >= 3
		AND AVG(input_tokens) >= 1000
		ORDER BY total_input_cost DESC`)
	if err != nil {
		http.Error(w, "failed to fetch optimizations", http.StatusInternalServerError)
		return
	}
	defer repeatedRows.Close()

	for repeatedRows.Next() {
		var summary models.OptimizationSummary
		err := repeatedRows.Scan(
			&summary.Feature,
			&summary.PromptHash,
			&summary.Requests,
			&summary.AvgInputTokens,
			&summary.TotalInputCost,
		)
		if err != nil {
			http.Error(w, "failed to scan optimization", http.StatusInternalServerError)
			return
		}
		summary.Type = "repeated_context"
		summary.Message = "This feature repeatedly sends the same large input context."
		summaries = append(summaries, summary)
	}
	if err := repeatedRows.Err(); err != nil {
		http.Error(w, "failed to read optimizations", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(summaries)
}
