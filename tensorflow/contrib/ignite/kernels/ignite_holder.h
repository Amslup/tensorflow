/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "ignite/ignite.h"
#include "ignite/ignition.h"

using namespace ignite;

class IgniteHolder {
public:
    static IgniteHolder &Instance() {
        static IgniteHolder* s = new IgniteHolder;
        return *s;
    }

    Ignite& getIgnite() {
        return ignite;
    }

private:
    Ignite ignite;
    IgniteHolder() {
        IgniteConfiguration cfg;
        std::string path(std::getenv("TF_IGNITE_CLIENT_CONFIG"));
        cfg.springCfgPath = path;
        ignite = Ignition::Start(cfg);
    }

    ~IgniteHolder() {
        Ignition::Stop(ignite.GetName(), false);
    }

    IgniteHolder(IgniteHolder const &);
    IgniteHolder &operator=(IgniteHolder const &);
};