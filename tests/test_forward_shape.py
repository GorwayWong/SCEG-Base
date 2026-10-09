"""Phase2 checks the actual 5-way/1-shot/15-query episode, including ranges."""
import torch
from models import SCEG


def test_forward_shape(model_config, config, device):
    settings = config["verification"]
    way, shot, query = (settings[key] for key in ("way", "shot", "query"))
    batch = way*(shot+query)
    model = SCEG(model_config).to(device).eval()
    images = torch.randn(batch, model_config.input_channels,
                         model_config.image_size, model_config.image_size, device=device)
    with torch.no_grad():
        chunks = [model.encode(chunk) for chunk in images.split(settings["microbatch"])]
        embeddings = tuple(torch.cat([chunk[stage] for chunk in chunks]) for stage in range(3))
        output = model.classify(embeddings, way, shot)
    for embedding in embeddings:
        assert embedding.shape == (batch, model_config.feature_dim)
        assert torch.isfinite(embedding).all()
    for cosine in output[:3]:
        assert cosine.shape == (way*query, way)
    assert output.global_cosine.shape == (batch, model_config.base_classes)
    for cosine in output:
        assert torch.isfinite(cosine).all()
        assert cosine.abs().max() <= 1+settings["range_tolerance"]
    assert output.scores.shape == (way*query, way)
    assert output.scores.abs().max() <= 3+settings["range_tolerance"]
    predictions = output.scores.argmax(dim=1)
    assert predictions.min() >= 0 and predictions.max() < way


def test_backbone_stage_shapes(model_config, device):
    model = SCEG(model_config).to(device).eval()
    with torch.no_grad():
        features = model.feature_extractor(torch.randn(
            1, model_config.input_channels, model_config.image_size,
            model_config.image_size, device=device))
    assert [tuple(feature.shape) for feature in features] == [
        (1, 160, 21, 21), (1, 320, 10, 10), (1, 640, 5, 5)]


def test_einsum_matches_independent_cosine(model_config, device):
    classifier = SCEG(model_config).to(device).classifier
    way, shot, query = 5, 1, 15
    embeddings = torch.randn(way*(shot+query), model_config.feature_dim, device=device)
    cosine = classifier(embeddings, way, shot)
    support = embeddings[:way*shot].reshape(way, shot, -1).mean(dim=1)
    query_features = embeddings[way*shot:]
    numerator = torch.nn.functional.linear(query_features, support)
    denominator = ((query_features.square().sum(1)+model_config.episode_norm_eps).sqrt()[:, None]
                   * (support.square().sum(1)+model_config.episode_norm_eps).sqrt()[None, :])
    torch.testing.assert_close(cosine, numerator/denominator)
