"""Narrow TLS client fix for the pinned SecretFlow 1.14 native gRPC backend.

Upstream _init_channel passes ServerCredentials to secure_channel. This adapter
uses ChannelCredentials and refuses an unencrypted fallback. Server setup stays
upstream, with mandatory client authentication whenever ca_cert is configured.
"""
from __future__ import annotations

from pathlib import Path


def initialize_tls_channels(proxy, grpc_module, stub_factory):
    config = proxy._tls_config
    if not config or set(config) != {"ca_cert", "cert", "key"}:
        raise ValueError("Native federation requires all mutual-TLS credentials")
    credentials = grpc_module.ssl_channel_credentials(
        root_certificates=Path(config["ca_cert"]).read_bytes(),
        private_key=Path(config["key"]).read_bytes(),
        certificate_chain=Path(config["cert"]).read_bytes())
    for party, address in proxy._addresses.items():
        channel = grpc_module.secure_channel(address, credentials, options=proxy._grpc_options)
        proxy._stubs[party] = stub_factory(channel)


def install_pinned_tls_adapter():
    from importlib.metadata import version
    import grpc
    from secretflow.distributed.fed.proxy.grpc.grpc import GrpcProxy, fed_pb2_grpc
    if version("secretflow") != "1.14.0b0":
        raise RuntimeError("TLS adapter is audited only for the pinned SecretFlow")
    def initialize(proxy):
        initialize_tls_channels(proxy, grpc, fed_pb2_grpc.SfFedProxyStub)
    GrpcProxy._init_channel = initialize


def reject_anonymous_client(address, ca_path, timeout):
    import grpc
    channel = grpc.secure_channel(address, grpc.ssl_channel_credentials(
        root_certificates=Path(ca_path).read_bytes()))
    try:
        grpc.channel_ready_future(channel).result(timeout=timeout)
    except grpc.FutureTimeoutError:
        return True
    finally:
        channel.close()
    return False


def reject_anonymous_spu(address, ca_path, timeout):
    import socket
    import ssl
    host, port = address.rsplit(":", 1)
    context = ssl.create_default_context(cafile=str(ca_path))
    context.maximum_version = ssl.TLSVersion.TLSv1_2
    try:
        with socket.create_connection((host, int(port)), timeout=timeout) as connection:
            with context.wrap_socket(connection, server_hostname=host):
                return False
    except ssl.SSLError as error:
        return "ALERT" in str(error).upper() or "HANDSHAKE_FAILURE" in str(error).upper()
