"""Scoped precision instrumentation must preserve ordinary provider ownership."""

from contextvars import Context

import pytest

from pyrpmd.parallel import execution_plan, task_potential
from pyrpmd.parallel.execution import _context, _task_potential_transform


class _Provider:
    metadata = {}

    def __init__(self):
        self.closed = 0
        self.binding = object()

    def native_binding(self):
        return self.binding

    def close(self):
        self.closed += 1


class _Borrowed:
    def __init__(self, provider, label):
        self.provider = provider
        self.label = label
        self.closed = 0

    def close(self):
        self.closed += 1


@pytest.fixture
def providers(monkeypatch):
    made = []

    def make(spec, system):
        provider = _Provider()
        made.append(provider)
        return provider

    monkeypatch.setattr("pyrpmd.parallel.execution.make_task_potential", make)
    return made


@pytest.mark.parametrize("reuse", [False, True])
def test_unscoped_leases_preserve_provider_identity_and_close_once(providers, reuse):
    with _context(execution_plan(), 0):
        for _ in range(3):
            with task_potential({"provider": "free"}, None, reuse=reuse) as got:
                assert got is providers[-1]
                assert got.native_binding() is providers[-1].binding
                assert got.closed == 0
        assert len(providers) == (1 if reuse else 3)
    assert all(provider.closed == 1 for provider in providers)


@pytest.mark.parametrize("interruption", [RuntimeError, KeyboardInterrupt])
def test_nested_scopes_restore_outer_then_unscoped_after_exception(providers, interruption):
    wrappers = []

    def wrap(label):
        def transform(provider):
            wrapper = _Borrowed(provider, label)
            wrappers.append(wrapper)
            return wrapper
        return transform

    with _context(execution_plan(), 0):
        with _task_potential_transform(wrap("outer")):
            with task_potential({"provider": "free"}, None) as outer:
                assert outer.label == "outer"
            with pytest.raises(interruption):
                with _task_potential_transform(wrap("inner")):
                    with task_potential({"provider": "free"}, None) as inner:
                        assert inner.label == "inner"
                        assert inner.provider is outer.provider
                    raise interruption("scope interrupted")
            with task_potential({"provider": "free"}, None) as restored:
                assert restored.label == "outer"
                assert restored.provider is outer.provider
            assert providers[0].closed == 0
        with task_potential({"provider": "free"}, None) as unscoped:
            assert unscoped is outer.provider
    assert len(providers) == 1 and providers[0].closed == 1
    assert all(wrapper.closed == 0 for wrapper in wrappers)


@pytest.mark.parametrize("reuse", [False, True])
def test_transform_failure_closes_original_and_does_not_leak_scope(providers, reuse):
    def fail(provider):
        raise RuntimeError("cannot instrument")

    with _context(execution_plan(), 0):
        with pytest.raises(RuntimeError, match="cannot instrument"):
            with _task_potential_transform(fail):
                with task_potential({"provider": "free"}, None, reuse=reuse):
                    pytest.fail("failed transform yielded a provider")
        assert providers[0].closed == 1
        with task_potential({"provider": "free"}, None, reuse=reuse) as fresh:
            assert fresh is providers[1]
    assert len(providers) == 2
    assert all(provider.closed == 1 for provider in providers)


def test_scope_is_local_to_the_current_context(providers):
    def lease():
        with task_potential({"provider": "free"}, None) as provider:
            return provider

    with _context(execution_plan(), 0):
        with _task_potential_transform(lambda provider: _Borrowed(provider, "current")):
            wrapped = lease()
            unwrapped = Context().run(lease)
            assert unwrapped is wrapped.provider
            assert lease().provider is unwrapped
    assert len(providers) == 1 and providers[0].closed == 1


def test_invalid_transform_fails_without_changing_context(providers):
    with pytest.raises(TypeError, match="must be callable"):
        with _task_potential_transform(None):
            pytest.fail("invalid transform entered")
    with task_potential({"provider": "free"}, None) as provider:
        assert provider is providers[0]
    assert providers[0].closed == 1
