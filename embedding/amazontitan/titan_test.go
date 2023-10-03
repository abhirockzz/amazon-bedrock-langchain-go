package amazontitan

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestEmbedQuery(t *testing.T) {
	titanEmbedder, err := New("us-east-1")

	assert.Nil(t, err)

	result, err := titanEmbedder.EmbedQuery(context.Background(), "foo")
	assert.Nil(t, err)

	assert.Equal(t, 1536, len(result))
}

func TestEmbedDocuments(t *testing.T) {

	titanEmbedder, err := New("us-east-1")
	assert.Nil(t, err)

	texts := []string{"foo", "barbaz"}

	titanEmbedder.BatchSize = 3
	result, err := titanEmbedder.EmbedDocuments(context.Background(), texts)
	assert.Nil(t, err)

	assert.Equal(t, 2, len(result))

	assert.Equal(t, 1536, len(result[0]))
	assert.Equal(t, 1536, len(result[1]))
}
