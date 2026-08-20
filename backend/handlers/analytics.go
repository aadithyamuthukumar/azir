package handlers

import (
	"context"
	"encoding/json"
	"net/http"

	"github.com/aadithyamuthukumar/azir/models"
	"github.com/jackc/pgx/v5"
)

func Summary(w http.ResponseWriter, r *http.Request, conn *pgx.Conn) {

	var summary models.AnalyticsSummary
	err := conn.QueryRow(
		context.Background(),
		`SELECT
			COUNT(*),
			COALESCE(SUM(input_tokens), 0),
			COALESCE(SUM(output_tokens), 0),
			COALESCE(SUM(total_cost), 0)
		FROM llm_events`,
	).Scan(
		&summary.TotalRequests,
		&summary.InputTokens,
		&summary.OutputTokens,
		&summary.TotalCost,
	)

	if err != nil {
		http.Error(w, "failed to fetch summary", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(summary)

}

func FeaturesAnalytics(w http.ResponseWriter, r *http.Request, conn *pgx.Conn) {
	rows, err := conn.Query(
		context.Background(),
		`SELECT
			feature,
			COUNT(*),
			COALESCE(SUM(total_cost), 0)
		FROM llm_events
		GROUP BY feature
		ORDER BY SUM(total_cost) DESC`)

	if err != nil {
		http.Error(w, "failed to fetch feature analytics", http.StatusInternalServerError)
		return
	}
	defer rows.Close()

	var summaries []models.FeatureAnalyticsSummary
	for rows.Next() {
		var summary models.FeatureAnalyticsSummary
		err := rows.Scan(
			&summary.Feature,
			&summary.Requests,
			&summary.TotalCost,
		)
		if err != nil {
			http.Error(w, "failed to scan feature analytics", http.StatusInternalServerError)
			return
		}
		summaries = append(summaries, summary)
	}

	if err := rows.Err(); err != nil {
		http.Error(w, "failed to read feature analytics", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(summaries)
}

func ModelsAnalytics(w http.ResponseWriter, r *http.Request, conn *pgx.Conn) {
	rows, err := conn.Query(
		context.Background(),
		`SELECT
			model,
			COUNT(*),
			COALESCE(SUM(total_cost), 0)
		FROM llm_events
		GROUP BY model
		ORDER BY SUM(total_cost) DESC`)
	if err != nil {
		http.Error(w, "failed to fetch model analytics", http.StatusInternalServerError)
		return
	}
	defer rows.Close()

	var summaries []models.ModelAnalyticsSummary
	for rows.Next() {
		var summary models.ModelAnalyticsSummary
		err := rows.Scan(
			&summary.Model,
			&summary.Requests,
			&summary.TotalCost,
		)
		if err != nil {
			http.Error(w, "failed to scan model analytics", http.StatusInternalServerError)
			return
		}
		summaries = append(summaries, summary)
	}
	if err := rows.Err(); err != nil {
		http.Error(w, "failed to read model analytics", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(summaries)
}
