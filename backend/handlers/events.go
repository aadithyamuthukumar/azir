package handlers

import (
	"context"
	"encoding/json"
	"net/http"

	"github.com/aadithyamuthukumar/azir/models"
	"github.com/aadithyamuthukumar/azir/pricing"
	"github.com/jackc/pgx/v5"
)

func Post(w http.ResponseWriter, r *http.Request, conn *pgx.Conn) {

	var response models.LLMEvent
	err := json.NewDecoder(r.Body).Decode(&response)
	if err != nil {
		http.Error(w, "invalid JSON", http.StatusBadRequest)
		return
	}

	price, exists := pricing.ForModel(response.Model)
	if !exists {
		http.Error(w, "unknown model", http.StatusBadRequest)
		return
	}

	cost := models.EventCost{
		InputCost:  float64(response.InputTokens) / 1000 * price.InputPrice,
		OutputCost: float64(response.OutputTokens) / 1000 * price.OutputPrice,
		TotalCost:  float64(response.InputTokens)/1000*price.InputPrice + float64(response.OutputTokens)/1000*price.OutputPrice,
	}

	_, err = conn.Exec(context.Background(), "INSERT INTO llm_events (model, feature, input_tokens, output_tokens, latency_ms, input_cost, output_cost, total_cost, prompt_hash) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)",
		response.Model, response.Feature, response.InputTokens, response.OutputTokens, response.LatencyMs, cost.InputCost, cost.OutputCost, cost.TotalCost, response.PromptHash)
	if err != nil {
		http.Error(w, "failed to insert event", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)

	json.NewEncoder(w).Encode(struct {
		Event models.LLMEvent  `json:"event"`
		Cost  models.EventCost `json:"cost"`
	}{
		Event: response,
		Cost:  cost,
	})

}
