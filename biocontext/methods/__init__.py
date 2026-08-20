from . import ant_context, baseline, biocontext, retina

METHOD_REGISTRY = {
    "baseline": baseline.match,
    "retina_only": retina.match,
    "context_only": ant_context.match,
    "biocontext": biocontext.match,
}

__all__ = ["METHOD_REGISTRY", "baseline", "retina", "ant_context", "biocontext"]
