package main

import (
	"context"
	"encoding/json"
	"log"
	"net/http"
	"os"

	"github.com/jackc/pgx/v5"
	"github.com/joho/godotenv"
)

type llmEvent struct {
	Model        string `json:"model"`
	Feature      string `json:"feature"`
	InputTokens  int    `json:"input_tokens"`
	OutputTokens int    `json:"output_tokens"`
	LatencyMs    int    `json:"latency_ms"`
}

type modelPrice struct {
	InputPrice  float64 `json:"input_price"`
	OutputPrice float64 `json:"output_price"`
}

type eventCost struct {
	InputCost  float64 `json:"input_cost"`
	OutputCost float64 `json:"output_cost"`
	TotalCost  float64 `json:"total_cost"`
}

var modelPrices = map[string]modelPrice{
	"gpt-5-mini": {
		InputPrice:  0.001,
		OutputPrice: 0.002,
	},
}

func healthHandler(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusOK)
	w.Write([]byte("OK"))
}

func postHandler(w http.ResponseWriter, r *http.Request) {
	var response llmEvent
	err := json.NewDecoder(r.Body).Decode(&response)
	if err != nil {
		http.Error(w, "invalid JSON", http.StatusBadRequest)
		return
	}

	price := modelPrices[response.Model]
	cost := eventCost{
		InputCost:  float64(response.InputTokens) / 1000 * price.InputPrice,
		OutputCost: float64(response.OutputTokens) / 1000 * price.OutputPrice,
		TotalCost:  float64(response.InputTokens)/1000*price.InputPrice + float64(response.OutputTokens)/1000*price.OutputPrice,
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(struct {
		Event llmEvent  `json:"event"`
		Cost  eventCost `json:"cost"`
	}{
		Event: response,
		Cost:  cost,
	})

}

func main() {
	err := godotenv.Load()
	if err != nil {
		log.Fatalf("Error loading .env file: %v", err)
	}

	http.HandleFunc("GET /health", healthHandler)

	http.HandleFunc("POST /v1/events", postHandler)

	databaseURL := os.Getenv("DATABASE_URL")

	conn, err := pgx.Connect(context.Background(), databaseURL)
	if err != nil {
		log.Fatalf("failed to connect to database: %v", err)
		return
	}
	defer conn.Close(context.Background())

	http.ListenAndServe(":8080", nil)

}
