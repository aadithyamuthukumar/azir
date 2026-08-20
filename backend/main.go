package main

import (
	"context"
	"log"
	"net/http"

	"github.com/aadithyamuthukumar/azir/db"
	"github.com/aadithyamuthukumar/azir/handlers"
	"github.com/joho/godotenv"
)

func main() {
	err := godotenv.Load()
	if err != nil {
		log.Fatalf("Error loading .env file: %v", err)
	}

	http.HandleFunc("GET /health", handlers.Health)

	conn, err := db.Connect()
	if err != nil {
		log.Fatalf("failed to connect to database: %v", err)
		return
	}
	defer conn.Close(context.Background())

	http.HandleFunc("POST /v1/events", func(w http.ResponseWriter, r *http.Request) {
		handlers.Post(w, r, conn)
	})

	http.HandleFunc("GET /v1/analytics/summary", func(w http.ResponseWriter, r *http.Request) {
		handlers.Summary(w, r, conn)
	})

	http.HandleFunc("GET /v1/analytics/features", func(w http.ResponseWriter, r *http.Request) {
		handlers.FeaturesAnalytics(w, r, conn)
	})

	http.HandleFunc("GET /v1/analytics/models", func(w http.ResponseWriter, r *http.Request) {
		handlers.ModelsAnalytics(w, r, conn)
	})

	http.HandleFunc("GET /v1/optimizations", func(w http.ResponseWriter, r *http.Request) {
		handlers.Optimizations(w, r, conn)
	})

	http.ListenAndServe(":8080", nil)
}
