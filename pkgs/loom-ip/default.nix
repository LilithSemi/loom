{
  flakever,
  lib,
  buildDartApplication,
  mkShell,
  dart,
  yq,
}:
buildDartApplication {
  pname = "loom-ip";
  inherit (flakever) version;

  src = lib.fileset.toSource {
    root = ../../ip;
    fileset = lib.fileset.unions [
      ../../ip/bin
      ../../ip/lib
      ../../ip/test
      ../../ip/pubspec.yaml
      ../../ip/pubspec.lock
      ../../ip/analysis_options.yaml
    ];
  };

  pubspecLock = lib.importJSON ../../ip/pubspec.lock.json;

  gitHashes = {
    harbor = "sha256-O+y5lb3cR3Ebm92NUTCgCvbJyLlJ0De/d9j5Jn5F8/U=";
  };

  dartEntryPoints."bin/loom-genip" = "bin/loom_genip.dart";

  doCheck = true;

  checkPhase = ''
    runHook preCheck

    export HOME=$TMPDIR
    export PUB_CACHE=$TMPDIR/.pub-cache

    testPkgRoot=$(jq --raw-output \
      '.packages[] | select(.name == "test") | .rootUri | sub("file://"; "")' \
      .dart_tool/package_config.json)

    dart --packages=.dart_tool/package_config.json \
      "$testPkgRoot/bin/test.dart"

    runHook postCheck
  '';

  passthru.shell = mkShell {
    name = "loom-ip-dev-shell";
    packages = [
      dart
      yq
    ];
  };
}
