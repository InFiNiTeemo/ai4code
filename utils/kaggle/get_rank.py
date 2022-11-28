import six
from kaggle.api.kaggle_api_extended import KaggleApi


class CustomKaggleApi(KaggleApi):
    """
    a custom api supporting sort_by_public_score when listing the submissions
    """
    def competitions_submissions_list_with_http_info(self, id, **kwargs):
        all_params = ['id', 'page', "sortBy"]  # noqa: E501
        all_params.append('async_req')
        all_params.append('_return_http_data_only')
        all_params.append('_preload_content')
        all_params.append('_request_timeout')

        params = locals()
        for key, val in six.iteritems(params['kwargs']):
            if key not in all_params:
                raise TypeError(
                    "Got an unexpected keyword argument '%s'"
                    " to method competitions_submissions_list" % key
                )
            params[key] = val
        del params['kwargs']
        # verify the required parameter 'id' is set
        if ('id' not in params or
                params['id'] is None):
            raise ValueError("Missing the required parameter `id` when calling `competitions_submissions_list`")  # noqa: E501

        collection_formats = {}

        path_params = {}
        if 'id' in params:
            path_params['id'] = params['id']  # noqa: E501

        query_params = []
        if 'page' in params:
            query_params.append(('page', params['page']))  # noqa: E501
        if 'sortBy' in params:
            query_params.append(('sortBy', params['sortBy']))

        header_params = {}

        form_params = []
        local_var_files = {}

        body_params = None
        # HTTP header `Accept`
        header_params['Accept'] = self.api_client.select_header_accept(
            ['application/json'])  # noqa: E501

        # Authentication setting
        auth_settings = ['basicAuth']  # noqa: E501

        # query_params.append(('sortBy', 'SUBMISSION_SORT_BY_PUBLIC_SCORE'))
        return self.api_client.call_api(
            '/competitions/submissions/list/{id}', 'GET',
            path_params,
            query_params,
            header_params,
            body=body_params,
            post_params=form_params,
            files=local_var_files,
            response_type='Result',  # noqa: E501
            auth_settings=auth_settings,
            async_req=params.get('async_req'),
            _return_http_data_only=params.get('_return_http_data_only'),
            _preload_content=params.get('_preload_content', True),
            _request_timeout=params.get('_request_timeout'),
            collection_formats=collection_formats
        )


api = CustomKaggleApi()
api.authenticate()
COMPETITION = 'feedback-prize-english-language-learning'

all_submissions = []
page = 1
while True:
    submissions, _, _ = api.competitions_submissions_list_with_http_info(
        id=COMPETITION,
        page=page,
        sortBy="SUBMISSION_SORT_BY_PUBLIC_SCORE",
    )
    if len(submissions) == 0:
        break
    all_submissions.extend(submissions)
    page += 1

all_submissions = [s for s in all_submissions if s["hasPublicScore"]]
desc2info = {}
for i, s in enumerate(all_submissions):
    if not s["hasDescription"]:
        continue
    s["innerRank"] = i + 1
    desc = s["description"].split('|')[0]
    desc2info.setdefault(desc, [])
    desc2info[desc].append(s)