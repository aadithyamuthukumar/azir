package pricing

import "github.com/aadithyamuthukumar/azir/models"

var modelPrices = map[string]models.ModelPrice{
	"gpt-5-mini": {
		InputPrice:  0.001,
		OutputPrice: 0.002,
	},
}

// ForModel returns the pricing for a model. The second return value is false
// when the model is unknown.
func ForModel(model string) (models.ModelPrice, bool) {
	price, exists := modelPrices[model]
	return price, exists
}
