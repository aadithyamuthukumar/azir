package db

import (
	"context"
	"os"

	"github.com/jackc/pgx/v5"
)

func Connect() (*pgx.Conn, error) {
	databaseURL := os.Getenv("DATABASE_URL")
	return pgx.Connect(context.Background(), databaseURL)
}
